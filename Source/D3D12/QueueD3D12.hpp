// © 2021 NVIDIA Corporation

Result QueueD3D12::Create(QueueType queueType, float priority) {
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Priority = priority > 0.5f ? D3D12_COMMAND_QUEUE_PRIORITY_HIGH : D3D12_COMMAND_QUEUE_PRIORITY_NORMAL; // TODO: values in between? check D3D12_FEATURE_COMMAND_QUEUE_PRIORITY support?
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.NodeMask = NODE_MASK;
    queueDesc.Type = GetCommandListType(queueType);

    HRESULT hr = m_Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_Queue));
    NRI_RETURN_ON_BAD_HRESULT(&m_Device, hr, "ID3D12Device::CreateQueue");

    m_CommandListType = queueDesc.Type;

    return Result::SUCCESS;
}

Result QueueD3D12::Create(ID3D12CommandQueue* queue) {
    if (!queue)
        return Result::INVALID_ARGUMENT;

    const D3D12_COMMAND_QUEUE_DESC& queueDesc = queue->GetDesc();

    m_Queue = queue;
    m_CommandListType = queueDesc.Type;

    return Result::SUCCESS;
}

NRI_INLINE void QueueD3D12::BeginAnnotation(const char* name, uint32_t bgra) {
    if (m_Device.HasPix())
        m_Device.GetPix().BeginEventOnQueue(m_Queue, bgra, name);
    else
        PIXBeginEvent(m_Queue, bgra, name);
}

NRI_INLINE void QueueD3D12::EndAnnotation() {
    if (m_Device.HasPix())
        m_Device.GetPix().EndEventOnQueue(m_Queue);
    else
        PIXEndEvent(m_Queue);
}

NRI_INLINE void QueueD3D12::Annotation(const char* name, uint32_t bgra) {
    if (m_Device.HasPix())
        m_Device.GetPix().SetMarkerOnQueue(m_Queue, bgra, name);
    else
        PIXSetMarker(m_Queue, bgra, name);
}

NRI_INLINE void QueueD3D12::GetCalibratedTimestamps(uint64_t& timestampGPU, uint64_t& timestampCPU) {
    HRESULT hr = m_Queue->GetClockCalibration(&timestampGPU, &timestampCPU);
    NRI_RETURN_VOID_ON_BAD_HRESULT(&m_Device, hr, "GetClockCalibration");
}

NRI_INLINE Result QueueD3D12::Submit(const QueueSubmitDesc& queueSubmitDesc) {
    for (uint32_t i = 0; i < queueSubmitDesc.waitFenceNum; i++) {
        const FenceSubmitDesc& fenceSubmitDesc = queueSubmitDesc.waitFences[i];
        FenceD3D12* fence = (FenceD3D12*)fenceSubmitDesc.fence;
        fence->QueueWait(*this, fenceSubmitDesc.value);
    }

    if (queueSubmitDesc.commandBufferNum) {
        Scratch<ID3D12CommandList*> commandLists = NRI_ALLOCATE_SCRATCH(m_Device, ID3D12CommandList*, queueSubmitDesc.commandBufferNum);
        for (uint32_t j = 0; j < queueSubmitDesc.commandBufferNum; j++)
            commandLists[j] = *(CommandBufferD3D12*)queueSubmitDesc.commandBuffers[j];

        m_Queue->ExecuteCommandLists(queueSubmitDesc.commandBufferNum, commandLists);
    }

    for (uint32_t i = 0; i < queueSubmitDesc.signalFenceNum; i++) {
        const FenceSubmitDesc& fenceSubmitDesc = queueSubmitDesc.signalFences[i];
        FenceD3D12* fence = (FenceD3D12*)fenceSubmitDesc.fence;
        fence->QueueSignal(*this, fenceSubmitDesc.value);
    }

    // Is device lost?
    HRESULT hr = m_Device->GetDeviceRemovedReason() == S_OK ? S_OK : DXGI_ERROR_DEVICE_REMOVED;
    NRI_RETURN_ON_BAD_HRESULT(&m_Device, hr, "Submit");

    return Result::SUCCESS;
}

struct TileMapping {
    ID3D12Resource* resource;
    ID3D12Heap* heap;
    D3D12_TILED_RESOURCE_COORDINATE coordinate;
    D3D12_TILE_REGION_SIZE region;
    UINT heapOffset;
    bool unmap;
};

static bool SetHeapTile(const MemoryD3D12* memory, uint64_t memoryOffset, uint32_t tileSize, UINT& heapOffset) {
    if (!memory)
        return false;

    const uint64_t byteOffset = memory->GetOffset() + memoryOffset;

    if (tileSize == 0 || (byteOffset % tileSize) != 0)
        return false;

    const uint64_t tileIndex = byteOffset / tileSize;

    if (tileIndex > UINT32_MAX)
        return false;

    heapOffset = (UINT)tileIndex;

    return true;
}

NRI_INLINE Result QueueD3D12::BindSparse(const SparseBufferMappingDesc* bufferMappings, uint32_t bufferMappingNum, const SparseTextureMappingDesc* textureMappings, uint32_t textureMappingNum) {
    if (bufferMappingNum == 0 && textureMappingNum == 0)
        return Result::SUCCESS;

    const uint32_t tileSize = m_Device.GetDesc().sparse.tileSize;
    const uint32_t mappingNum = bufferMappingNum + textureMappingNum;
    Scratch<TileMapping> mappings = NRI_ALLOCATE_SCRATCH(m_Device, TileMapping, mappingNum);
    uint32_t count = 0;

    for (uint32_t i = 0; i < bufferMappingNum; i++) {
        const SparseBufferMappingDesc& mapping = bufferMappings[i];
        BufferD3D12& buffer = *(BufferD3D12*)mapping.buffer;
        const bool unmap = mapping.mapping == SparseMapping::UNMAP;

        if (tileSize == 0 || (mapping.offset % tileSize) != 0 || (mapping.size % tileSize) != 0 || mapping.size == 0)
            return Result::INVALID_ARGUMENT;

        const uint64_t tileIndex = mapping.offset / tileSize;
        const uint64_t tileCount = mapping.size / tileSize;

        if (tileIndex > UINT32_MAX || tileCount > UINT32_MAX)
            return Result::FAILURE;

        TileMapping& out = mappings[count++];
        out = {};
        out.resource = buffer;
        out.unmap = unmap;
        out.coordinate.X = (UINT)tileIndex;
        out.coordinate.Y = 0;
        out.coordinate.Z = 0;
        out.coordinate.Subresource = 0;
        out.region.NumTiles = (UINT)tileCount;
        out.region.UseBox = FALSE;
        out.region.Width = 0;
        out.region.Height = 0;
        out.region.Depth = 0;
        out.heap = nullptr;
        out.heapOffset = 0;

        if (!unmap) {
            if (!SetHeapTile((MemoryD3D12*)mapping.memory, mapping.memoryOffset, tileSize, out.heapOffset))
                return Result::FAILURE;

            out.heap = *(MemoryD3D12*)mapping.memory;
        }
    }

    for (uint32_t i = 0; i < textureMappingNum; i++) {
        const SparseTextureMappingDesc& mapping = textureMappings[i];
        TextureD3D12& texture = *(TextureD3D12*)mapping.texture;
        const TextureDesc& textureDesc = texture.GetDesc();
        const TextureTilingDesc& tiling = texture.GetTiling();
        const bool unmap = mapping.mapping == SparseMapping::UNMAP;

        if (mapping.metadata)
            return Result::UNSUPPORTED;

        const bool packedTail = IsSparsePackedTail(textureDesc, tiling, mapping.region);

        TileMapping& out = mappings[count++];
        out = {};
        out.resource = texture;
        out.unmap = unmap;
        out.heap = nullptr;
        out.heapOffset = 0;

        if (packedTail) {
            const Dim_t firstPackedMip = (Dim_t)(textureDesc.mipNum - tiling.packedMipNum);
            out.coordinate.X = 0;
            out.coordinate.Y = 0;
            out.coordinate.Z = 0;
            out.coordinate.Subresource = (UINT)firstPackedMip + (UINT)mapping.region.layerOffset * (UINT)textureDesc.mipNum;
            out.region.NumTiles = tiling.packedTileNum;
            out.region.UseBox = FALSE;
            out.region.Width = 0;
            out.region.Height = 0;
            out.region.Depth = 0;
        } else {
            const TextureRegionDesc expanded = ExpandWholeSizeRegion(GraphicsAPI::D3D12, textureDesc, mapping.region);

            if (tiling.tileWidth == 0 || tiling.tileHeight == 0 || tiling.tileDepth == 0)
                return Result::FAILURE;

            const uint64_t tilesX = ((uint64_t)expanded.width + tiling.tileWidth - 1) / tiling.tileWidth;
            const uint64_t tilesY = ((uint64_t)expanded.height + tiling.tileHeight - 1) / tiling.tileHeight;
            const uint64_t tilesZ = ((uint64_t)expanded.depth + tiling.tileDepth - 1) / tiling.tileDepth;

            if (tilesX == 0 || tilesY == 0 || tilesZ == 0 || tilesX > UINT32_MAX || tilesY > UINT32_MAX || tilesZ > UINT32_MAX)
                return Result::FAILURE;

            if (tilesX > UINT64_MAX / tilesY || (tilesX * tilesY) > UINT64_MAX / tilesZ)
                return Result::FAILURE;

            const uint64_t tiles = tilesX * tilesY * tilesZ;

            if (tiles > UINT32_MAX)
                return Result::FAILURE;

            out.coordinate.X = expanded.x / tiling.tileWidth;
            out.coordinate.Y = expanded.y / tiling.tileHeight;
            out.coordinate.Z = expanded.z / tiling.tileDepth;
            out.coordinate.Subresource = (UINT)mapping.region.mipOffset + (UINT)mapping.region.layerOffset * (UINT)textureDesc.mipNum;
            out.region.NumTiles = (UINT)tiles;
            out.region.UseBox = TRUE;
            out.region.Width = (UINT)tilesX;
            out.region.Height = (UINT)tilesY;
            out.region.Depth = (UINT)tilesZ;
        }

        if (out.region.NumTiles == 0)
            return Result::FAILURE;

        if (!unmap) {
            if (!SetHeapTile((MemoryD3D12*)mapping.memory, mapping.memoryOffset, tileSize, out.heapOffset))
                return Result::FAILURE;

            out.heap = *(MemoryD3D12*)mapping.memory;
        }
    }

    Scratch<D3D12_TILED_RESOURCE_COORDINATE> coordinates = NRI_ALLOCATE_SCRATCH(m_Device, D3D12_TILED_RESOURCE_COORDINATE, count);
    Scratch<D3D12_TILE_REGION_SIZE> regions = NRI_ALLOCATE_SCRATCH(m_Device, D3D12_TILE_REGION_SIZE, count);
    Scratch<D3D12_TILE_RANGE_FLAGS> rangeFlags = NRI_ALLOCATE_SCRATCH(m_Device, D3D12_TILE_RANGE_FLAGS, count);
    Scratch<UINT> heapOffsets = NRI_ALLOCATE_SCRATCH(m_Device, UINT, count);
    Scratch<UINT> rangeTileCounts = NRI_ALLOCATE_SCRATCH(m_Device, UINT, count);

    uint32_t cursor = 0;

    while (cursor < count) {
        const uint32_t start = cursor;
        ID3D12Resource* resource = mappings[start].resource;
        ID3D12Heap* heap = mappings[start].heap;
        const bool unmap = mappings[start].unmap;
        cursor++;

        while (cursor < count && mappings[cursor].resource == resource && mappings[cursor].heap == heap && mappings[cursor].unmap == unmap)
            cursor++;

        const uint32_t regionCount = cursor - start;

        for (uint32_t i = 0; i < regionCount; i++) {
            const TileMapping& mapping = mappings[start + i];
            coordinates[i] = mapping.coordinate;
            regions[i] = mapping.region;
            rangeFlags[i] = unmap ? D3D12_TILE_RANGE_FLAG_NULL : D3D12_TILE_RANGE_FLAG_NONE;
            heapOffsets[i] = mapping.heapOffset;
            rangeTileCounts[i] = mapping.region.NumTiles;
        }

        m_Queue->UpdateTileMappings(resource, regionCount, coordinates, regions, unmap ? nullptr : heap, regionCount, rangeFlags, unmap ? nullptr : (const UINT*)heapOffsets, rangeTileCounts, D3D12_TILE_MAPPING_FLAG_NONE);
    }

    const HRESULT hr = m_Device->GetDeviceRemovedReason() == S_OK ? S_OK : DXGI_ERROR_DEVICE_REMOVED;
    NRI_RETURN_ON_BAD_HRESULT(&m_Device, hr, "UpdateTileMappings");

    return Result::SUCCESS;
}

NRI_INLINE Result QueueD3D12::WaitIdle() {
    FenceD3D12* fence = nullptr;
    Result result = m_Device.CreateImplementation<FenceD3D12>(fence, 0);
    if (result == Result::SUCCESS) {
        fence->QueueSignal(*this, 1);
        result = fence->Wait(1);

        Destroy(fence);
    }

    return result;
}

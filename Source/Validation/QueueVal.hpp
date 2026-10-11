// © 2021 NVIDIA Corporation

NRI_INLINE void QueueVal::BeginAnnotation(const char* name, uint32_t bgra) {
    GetCoreInterfaceImpl().QueueBeginAnnotation(*GetImpl(), name, bgra);
}

NRI_INLINE void QueueVal::EndAnnotation() {
    GetCoreInterfaceImpl().QueueEndAnnotation(*GetImpl());
}

NRI_INLINE void QueueVal::Annotation(const char* name, uint32_t bgra) {
    GetCoreInterfaceImpl().QueueAnnotation(*GetImpl(), name, bgra);
}

NRI_INLINE void QueueVal::GetCalibratedTimestamps(uint64_t& timestampGPU, uint64_t& timestampCPU) {
    const DeviceDesc& deviceDesc = m_Device.GetDesc();
    NRI_RETURN_ON_FAILURE(&m_Device, deviceDesc.features.calibratedTimestamps, ReturnVoid(), "'features.calibratedTimestamps' is false");

    GetCoreInterfaceImpl().GetCalibratedTimestamps(*GetImpl(), timestampGPU, timestampCPU);
}

NRI_INLINE Result QueueVal::Submit(const QueueSubmitDesc& queueSubmitDesc) {
    NRI_RETURN_ON_FAILURE(&m_Device, !queueSubmitDesc.swapChain || queueSubmitDesc.presentId != 0, Result::INVALID_ARGUMENT, "'presentId' is 0");

    auto queueSubmitDescImpl = queueSubmitDesc;

    Scratch<FenceSubmitDesc> waitFences = NRI_ALLOCATE_SCRATCH(m_Device, FenceSubmitDesc, queueSubmitDesc.waitFenceNum);
    for (uint32_t i = 0; i < queueSubmitDesc.waitFenceNum; i++) {
        waitFences[i] = queueSubmitDesc.waitFences[i];
        waitFences[i].fence = NRI_GET_IMPL(Fence, waitFences[i].fence);
    }
    queueSubmitDescImpl.waitFences = waitFences;

    Scratch<CommandBuffer*> commandBuffers = NRI_ALLOCATE_SCRATCH(m_Device, CommandBuffer*, queueSubmitDesc.commandBufferNum);
    for (uint32_t i = 0; i < queueSubmitDesc.commandBufferNum; i++)
        commandBuffers[i] = NRI_GET_IMPL(CommandBuffer, queueSubmitDesc.commandBuffers[i]);
    queueSubmitDescImpl.commandBuffers = commandBuffers;

    Scratch<FenceSubmitDesc> signalFences = NRI_ALLOCATE_SCRATCH(m_Device, FenceSubmitDesc, queueSubmitDesc.signalFenceNum);
    for (uint32_t i = 0; i < queueSubmitDesc.signalFenceNum; i++) {
        signalFences[i] = queueSubmitDesc.signalFences[i];
        signalFences[i].fence = NRI_GET_IMPL(Fence, signalFences[i].fence);
    }
    queueSubmitDescImpl.signalFences = signalFences;

    queueSubmitDescImpl.swapChain = NRI_GET_IMPL(SwapChain, queueSubmitDesc.swapChain);

    return GetCoreInterfaceImpl().QueueSubmit(*GetImpl(), queueSubmitDescImpl);
}

static bool MultiplyTiles(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > UINT64_MAX / a)
        return false;

    out = a * b;

    return true;
}

static Result ValidateSparseMemoryRange(DeviceVal& device, const MemoryVal* memory, uint64_t memoryOffset, uint64_t byteSize, uint32_t tileSize, uint32_t index) {
    NRI_RETURN_ON_FAILURE(&device, memory != nullptr, Result::INVALID_ARGUMENT, "'[%u].memory' is NULL", index);
    NRI_RETURN_ON_FAILURE(&device, byteSize != 0 && memoryOffset <= (UINT64_MAX - byteSize), Result::INVALID_ARGUMENT, "'[%u].memoryOffset' is out of range", index);
    NRI_RETURN_ON_FAILURE(&device, (memoryOffset % tileSize) == 0, Result::INVALID_ARGUMENT, "'[%u].memoryOffset' is misaligned", index);

    if (!memory->IsWrapped() && memory->GetSize() != 0)
        NRI_RETURN_ON_FAILURE(&device, (memoryOffset + byteSize) <= memory->GetSize(), Result::INVALID_ARGUMENT, "'[%u].memoryOffset' is out of range", index);

    return Result::SUCCESS;
}

NRI_INLINE Result QueueVal::BindSparse(const SparseBufferMappingDesc* bufferMappings, uint32_t bufferMappingNum, const SparseTextureMappingDesc* textureMappings, uint32_t textureMappingNum) {
    if (bufferMappingNum == 0 && textureMappingNum == 0)
        return Result::SUCCESS;

    const DeviceDesc& deviceDesc = m_Device.GetDesc();

    if (bufferMappingNum != 0) {
        NRI_RETURN_ON_FAILURE(&m_Device, bufferMappings != nullptr, Result::INVALID_ARGUMENT, "'bufferMappings' is NULL");
        NRI_RETURN_ON_FAILURE(&m_Device, deviceDesc.features.sparseBuffer, Result::UNSUPPORTED, "'features.sparseBuffer' is false");
    }

    if (textureMappingNum != 0) {
        NRI_RETURN_ON_FAILURE(&m_Device, textureMappings != nullptr, Result::INVALID_ARGUMENT, "'textureMappings' is NULL");
        NRI_RETURN_ON_FAILURE(&m_Device, deviceDesc.features.sparseTexture, Result::UNSUPPORTED, "'features.sparseTexture' is false");
    }

    const uint32_t tileSize = deviceDesc.sparse.tileSize;
    NRI_RETURN_ON_FAILURE(&m_Device, tileSize != 0, Result::UNSUPPORTED, "'sparse.tileSize' is 0");

    Scratch<SparseBufferMappingDesc> bufferMappingsImpl = NRI_ALLOCATE_SCRATCH(m_Device, SparseBufferMappingDesc, bufferMappingNum);

    for (uint32_t i = 0; i < bufferMappingNum; i++) {
        const SparseBufferMappingDesc& mapping = bufferMappings[i];

        NRI_RETURN_ON_FAILURE(&m_Device, mapping.buffer != nullptr, Result::INVALID_ARGUMENT, "'bufferMappings[%u].buffer' is NULL", i);
        NRI_RETURN_ON_FAILURE(&m_Device, mapping.mapping < SparseMapping::MAX_NUM, Result::INVALID_ARGUMENT, "'bufferMappings[%u].mapping' is invalid", i);

        const BufferVal& bufferVal = *(const BufferVal*)mapping.buffer;
        const BufferDesc& bufferDesc = bufferVal.GetDesc();

        NRI_RETURN_ON_FAILURE(&m_Device, bufferDesc.usage & BufferUsageBits::SPARSE, Result::INVALID_ARGUMENT, "'bufferMappings[%u].buffer' usage does not include 'BufferUsageBits::SPARSE'", i);
        NRI_RETURN_ON_FAILURE(&m_Device, mapping.offset <= bufferDesc.size, Result::INVALID_ARGUMENT, "'bufferMappings[%u].offset' is out of range", i);
        NRI_RETURN_ON_FAILURE(&m_Device, mapping.size != 0 && mapping.size <= (bufferDesc.size - mapping.offset), Result::INVALID_ARGUMENT, "'bufferMappings[%u].size' is out of range", i);
        NRI_RETURN_ON_FAILURE(&m_Device, (mapping.offset % tileSize) == 0 && (mapping.size % tileSize) == 0, Result::INVALID_ARGUMENT, "'bufferMappings[%u]' is not tile aligned", i);

        if (mapping.mapping == SparseMapping::BIND) {
            const Result memoryResult = ValidateSparseMemoryRange(m_Device, mapping.memory ? (const MemoryVal*)mapping.memory : nullptr, mapping.memoryOffset, mapping.size, tileSize, i);

            if (memoryResult != Result::SUCCESS)
                return memoryResult;
        }

        SparseBufferMappingDesc& mappingImpl = bufferMappingsImpl[i];
        mappingImpl = mapping;
        mappingImpl.buffer = bufferVal.GetImpl();
        mappingImpl.memory = NRI_GET_IMPL(Memory, mapping.memory);
    }

    Scratch<SparseTextureMappingDesc> textureMappingsImpl = NRI_ALLOCATE_SCRATCH(m_Device, SparseTextureMappingDesc, textureMappingNum);

    for (uint32_t i = 0; i < textureMappingNum; i++) {
        const SparseTextureMappingDesc& mapping = textureMappings[i];

        NRI_RETURN_ON_FAILURE(&m_Device, mapping.texture != nullptr, Result::INVALID_ARGUMENT, "'textureMappings[%u].texture' is NULL", i);
        NRI_RETURN_ON_FAILURE(&m_Device, mapping.mapping < SparseMapping::MAX_NUM, Result::INVALID_ARGUMENT, "'textureMappings[%u].mapping' is invalid", i);

        const TextureVal& textureVal = *(const TextureVal*)mapping.texture;
        const TextureDesc& textureDesc = textureVal.GetDesc();

        NRI_RETURN_ON_FAILURE(&m_Device, textureDesc.usage & TextureUsageBits::SPARSE, Result::INVALID_ARGUMENT, "'textureMappings[%u].texture' usage does not include 'TextureUsageBits::SPARSE'", i);

        TextureTilingDesc tiling = {};
        GetCoreInterfaceImpl().GetTextureTiling(*textureVal.GetImpl(), tiling);

        NRI_RETURN_ON_FAILURE(&m_Device, tiling.tileWidth != 0 && tiling.tileHeight != 0 && tiling.tileDepth != 0, Result::INVALID_ARGUMENT, "'textureMappings[%u].texture' has no tile shape", i);
        NRI_RETURN_ON_FAILURE(&m_Device, tiling.packedMipNum <= textureDesc.mipNum, Result::INVALID_ARGUMENT, "'textureMappings[%u].texture' packed mip count is invalid", i);

        uint64_t byteSize = 0;

        if (mapping.metadata) {
            NRI_RETURN_ON_FAILURE(&m_Device, tiling.metadataSize != 0, Result::INVALID_ARGUMENT, "'textureMappings[%u].metadata' is set, but 'metadataSize' is 0", i);
            byteSize = tiling.metadataSize;
        } else if (IsSparsePackedTail(textureDesc, tiling, mapping.region)) {
            if (tiling.singlePackedTail) {
                NRI_RETURN_ON_FAILURE(&m_Device, mapping.region.layerOffset == 0, Result::INVALID_ARGUMENT, "'textureMappings[%u].region.layerOffset' must be 0 for a single packed tail", i);
            } else {
                NRI_RETURN_ON_FAILURE(&m_Device, mapping.region.layerOffset < textureDesc.layerNum, Result::INVALID_ARGUMENT, "'textureMappings[%u].region.layerOffset' is out of range", i);
            }

            if (!MultiplyTiles(tiling.packedTileNum, tileSize, byteSize) || byteSize == 0)
                return Result::INVALID_ARGUMENT;
        } else {
            const Dim_t firstPackedMip = tiling.packedMipNum == 0 ? textureDesc.mipNum : (Dim_t)(textureDesc.mipNum - tiling.packedMipNum);
            NRI_RETURN_ON_FAILURE(&m_Device, mapping.region.mipOffset < firstPackedMip, Result::INVALID_ARGUMENT, "'textureMappings[%u].region.mipOffset' is not a standard mip", i);

            if (textureDesc.type == TextureType::TEXTURE_3D) {
                NRI_RETURN_ON_FAILURE(&m_Device, mapping.region.layerOffset == 0, Result::INVALID_ARGUMENT, "'textureMappings[%u].region.layerOffset' must be 0 for a 3D texture", i);
            } else {
                NRI_RETURN_ON_FAILURE(&m_Device, mapping.region.layerOffset < textureDesc.layerNum, Result::INVALID_ARGUMENT, "'textureMappings[%u].region.layerOffset' is out of range", i);
            }

            const TextureRegionDesc expanded = ExpandWholeSizeRegion(deviceDesc.graphicsAPI, textureDesc, mapping.region);
            const uint32_t mipWidth = GetDimension(deviceDesc.graphicsAPI, textureDesc, 0, mapping.region.mipOffset);
            const uint32_t mipHeight = GetDimension(deviceDesc.graphicsAPI, textureDesc, 1, mapping.region.mipOffset);
            const uint32_t mipDepth = GetDimension(deviceDesc.graphicsAPI, textureDesc, 2, mapping.region.mipOffset);
            const uint32_t xEnd = (uint32_t)expanded.x + expanded.width;
            const uint32_t yEnd = (uint32_t)expanded.y + expanded.height;
            const uint32_t zEnd = (uint32_t)expanded.z + expanded.depth;

            NRI_RETURN_ON_FAILURE(&m_Device, expanded.width != 0 && expanded.height != 0 && expanded.depth != 0, Result::INVALID_ARGUMENT, "'textureMappings[%u].region' extent is 0", i);
            NRI_RETURN_ON_FAILURE(&m_Device, (expanded.x % tiling.tileWidth) == 0 && (expanded.y % tiling.tileHeight) == 0 && (expanded.z % tiling.tileDepth) == 0, Result::INVALID_ARGUMENT, "'textureMappings[%u].region' offset is not tile aligned", i);
            NRI_RETURN_ON_FAILURE(&m_Device, xEnd <= mipWidth && yEnd <= mipHeight && zEnd <= mipDepth, Result::INVALID_ARGUMENT, "'textureMappings[%u].region' is outside the mip", i);
            NRI_RETURN_ON_FAILURE(&m_Device, (expanded.width % tiling.tileWidth) == 0 || xEnd == mipWidth, Result::INVALID_ARGUMENT, "'textureMappings[%u].region.width' is not tile aligned and does not reach the mip edge", i);
            NRI_RETURN_ON_FAILURE(&m_Device, (expanded.height % tiling.tileHeight) == 0 || yEnd == mipHeight, Result::INVALID_ARGUMENT, "'textureMappings[%u].region.height' is not tile aligned and does not reach the mip edge", i);
            NRI_RETURN_ON_FAILURE(&m_Device, (expanded.depth % tiling.tileDepth) == 0 || zEnd == mipDepth, Result::INVALID_ARGUMENT, "'textureMappings[%u].region.depth' is not tile aligned and does not reach the mip edge", i);

            const uint64_t tilesX = ((uint64_t)expanded.width + tiling.tileWidth - 1) / tiling.tileWidth;
            const uint64_t tilesY = ((uint64_t)expanded.height + tiling.tileHeight - 1) / tiling.tileHeight;
            const uint64_t tilesZ = ((uint64_t)expanded.depth + tiling.tileDepth - 1) / tiling.tileDepth;
            uint64_t tiles = 0;

            if (!MultiplyTiles(tilesX, tilesY, tiles) || !MultiplyTiles(tiles, tilesZ, tiles) || !MultiplyTiles(tiles, tileSize, byteSize) || byteSize == 0)
                return Result::INVALID_ARGUMENT;
        }

        if (mapping.mapping == SparseMapping::BIND) {
            const Result memoryResult = ValidateSparseMemoryRange(m_Device, mapping.memory ? (const MemoryVal*)mapping.memory : nullptr, mapping.memoryOffset, byteSize, tileSize, i);

            if (memoryResult != Result::SUCCESS)
                return memoryResult;
        }

        SparseTextureMappingDesc& mappingImpl = textureMappingsImpl[i];
        mappingImpl = mapping;
        mappingImpl.texture = textureVal.GetImpl();
        mappingImpl.memory = NRI_GET_IMPL(Memory, mapping.memory);
    }

    return GetCoreInterfaceImpl().QueueBindSparse(*GetImpl(), bufferMappingsImpl, bufferMappingNum, textureMappingsImpl, textureMappingNum);
}

NRI_INLINE Result QueueVal::WaitIdle() {
    return GetCoreInterfaceImpl().QueueWaitIdle(GetImpl());
}

// © 2021 NVIDIA Corporation

Result QueueVK::Create(QueueType type, uint32_t familyIndex, VkQueue handle, Lock* sharedLock, bool familySparse) {
    m_Type = type;
    m_FamilyIndex = familyIndex;
    m_Handle = handle;
    m_Lock = sharedLock ? sharedLock : &m_OwnLock;
    m_FamilySparse = familySparse;

    return Result::SUCCESS;
}

NRI_INLINE void QueueVK::SetDebugName(const char* name) {
    ExclusiveScope lock(*m_Lock);

    m_Device.SetDebugNameToTrivialObject(VK_OBJECT_TYPE_QUEUE, (uint64_t)m_Handle, name);
}

NRI_INLINE void QueueVK::BeginAnnotation(const char* name, uint32_t bgra) {
    ExclusiveScope lock(*m_Lock);

    VkDebugUtilsLabelEXT info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    info.pLabelName = name;
    info.color[0] = ((bgra >> 16) & 0xFF) / 255.0f;
    info.color[1] = ((bgra >> 8) & 0xFF) / 255.0f;
    info.color[2] = ((bgra >> 0) & 0xFF) / 255.0f;
    info.color[3] = 1.0f; // PIX sets alpha to 1

    const auto& vk = m_Device.GetDispatchTable();
    if (vk.QueueBeginDebugUtilsLabelEXT)
        vk.QueueBeginDebugUtilsLabelEXT(m_Handle, &info);
}

NRI_INLINE void QueueVK::EndAnnotation() {
    ExclusiveScope lock(*m_Lock);

    const auto& vk = m_Device.GetDispatchTable();
    if (vk.QueueEndDebugUtilsLabelEXT)
        vk.QueueEndDebugUtilsLabelEXT(m_Handle);
}

NRI_INLINE void QueueVK::Annotation(const char* name, uint32_t bgra) {
    ExclusiveScope lock(*m_Lock);

    VkDebugUtilsLabelEXT info = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    info.pLabelName = name;
    info.color[0] = ((bgra >> 16) & 0xFF) / 255.0f;
    info.color[1] = ((bgra >> 8) & 0xFF) / 255.0f;
    info.color[2] = ((bgra >> 0) & 0xFF) / 255.0f;
    info.color[3] = 1.0f; // PIX sets alpha to 1

    const auto& vk = m_Device.GetDispatchTable();
    if (vk.QueueInsertDebugUtilsLabelEXT)
        vk.QueueInsertDebugUtilsLabelEXT(m_Handle, &info);
}

NRI_INLINE void QueueVK::GetCalibratedTimestamps(uint64_t& timestampGPU, uint64_t& timestampCPU) {
    timestampGPU = 0;
    timestampCPU = 0;

    VkCalibratedTimestampInfoKHR timestampInfos[2] = {};
    {
        // GPU
        timestampInfos[0].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR;
        timestampInfos[0].timeDomain = VK_TIME_DOMAIN_DEVICE_KHR;

        // CPU
        timestampInfos[1].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR;
        timestampInfos[1].timeDomain = m_Device.GetCalibratedTimestampCPUTimeDomain();
    }

    uint64_t timestamps[2] = {};
    uint64_t maxDeviation = 0;

    const auto& vk = m_Device.GetDispatchTable();
    VkResult vkResult = vk.GetCalibratedTimestampsEXT(m_Device, 2, timestampInfos, timestamps, &maxDeviation);
    NRI_RETURN_VOID_ON_BAD_VKRESULT(&m_Device, vkResult, "GetCalibratedTimestampsKHR");

    timestampGPU = timestamps[0];
    timestampCPU = timestamps[1];
}

NRI_INLINE Result QueueVK::Submit(const QueueSubmitDesc& queueSubmitDesc) {
    ExclusiveScope lock(*m_Lock);

    Scratch<VkSemaphoreSubmitInfo> waitSemaphores = NRI_ALLOCATE_SCRATCH(m_Device, VkSemaphoreSubmitInfo, queueSubmitDesc.waitFenceNum);
    for (uint32_t i = 0; i < queueSubmitDesc.waitFenceNum; i++) {
        waitSemaphores[i] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        waitSemaphores[i].semaphore = *(FenceVK*)queueSubmitDesc.waitFences[i].fence;
        waitSemaphores[i].value = queueSubmitDesc.waitFences[i].value;
        waitSemaphores[i].stageMask = GetPipelineStageFlags(queueSubmitDesc.waitFences[i].stages);
    }

    Scratch<VkCommandBufferSubmitInfo> commandBuffers = NRI_ALLOCATE_SCRATCH(m_Device, VkCommandBufferSubmitInfo, queueSubmitDesc.commandBufferNum);
    for (uint32_t i = 0; i < queueSubmitDesc.commandBufferNum; i++) {
        commandBuffers[i] = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        commandBuffers[i].commandBuffer = *(CommandBufferVK*)queueSubmitDesc.commandBuffers[i];
    }

    Scratch<VkSemaphoreSubmitInfo> signalSemaphores = NRI_ALLOCATE_SCRATCH(m_Device, VkSemaphoreSubmitInfo, queueSubmitDesc.signalFenceNum);
    for (uint32_t i = 0; i < queueSubmitDesc.signalFenceNum; i++) {
        signalSemaphores[i] = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signalSemaphores[i].semaphore = *(FenceVK*)queueSubmitDesc.signalFences[i].fence;
        signalSemaphores[i].value = queueSubmitDesc.signalFences[i].value;
        signalSemaphores[i].stageMask = GetPipelineStageFlags(queueSubmitDesc.signalFences[i].stages);
    }

    VkSubmitInfo2 submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submitInfo.waitSemaphoreInfoCount = queueSubmitDesc.waitFenceNum;
    submitInfo.pWaitSemaphoreInfos = waitSemaphores;
    submitInfo.commandBufferInfoCount = queueSubmitDesc.commandBufferNum;
    submitInfo.pCommandBufferInfos = commandBuffers;
    submitInfo.signalSemaphoreInfoCount = queueSubmitDesc.signalFenceNum;
    submitInfo.pSignalSemaphoreInfos = signalSemaphores;

    VkLatencySubmissionPresentIdNV presentId = {VK_STRUCTURE_TYPE_LATENCY_SUBMISSION_PRESENT_ID_NV};
    if (queueSubmitDesc.swapChain && m_Device.m_IsSupported.presentId) {
        presentId.presentID = queueSubmitDesc.presentId;
        submitInfo.pNext = &presentId;
    }

    const auto& vk = m_Device.GetDispatchTable();
    VkResult vkResult = vk.QueueSubmit2(m_Handle, 1, &submitInfo, VK_NULL_HANDLE);
    NRI_RETURN_ON_BAD_VKRESULT(&m_Device, vkResult, "QueueSubmit2");

    return Result::SUCCESS;
}

static uint32_t AppendSparseAspects(VkImageAspectFlags aspectMask, const VkSparseImageMemoryBind& prototype, VkSparseImageMemoryBind* binds, uint32_t bindNum) {
    const VkImageAspectFlags aspects[] = {
        VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_ASPECT_STENCIL_BIT,
    };

    for (uint32_t i = 0; i < GetCountOf(aspects); i++) {
        if (aspectMask & aspects[i]) {
            binds[bindNum] = prototype;
            binds[bindNum].subresource.aspectMask = aspects[i];
            bindNum++;
        }
    }

    return bindNum;
}

NRI_INLINE Result QueueVK::BindSparse(const SparseBufferMappingDesc* bufferMappings, uint32_t bufferMappingNum, const SparseTextureMappingDesc* textureMappings, uint32_t textureMappingNum) {
    if (bufferMappingNum == 0 && textureMappingNum == 0)
        return Result::SUCCESS;

    if (!m_FamilySparse)
        return Result::UNSUPPORTED;

    uint32_t opaqueCapacity = textureMappingNum;

    for (uint32_t i = 0; i < textureMappingNum; i++) {
        if (!textureMappings[i].metadata)
            continue;

        const TextureVK& texture = *(const TextureVK*)textureMappings[i].texture;
        const uint32_t layers = texture.GetMetadataLayerNum();

        if (layers == 0)
            return Result::FAILURE;

        if (layers > 1 && opaqueCapacity > (UINT32_MAX - (layers - 1)))
            return Result::FAILURE;

        if (layers > 1)
            opaqueCapacity += layers - 1;
    }

    ExclusiveScope lock(*m_Lock);

    Scratch<VkSparseMemoryBind> bufferBinds = NRI_ALLOCATE_SCRATCH(m_Device, VkSparseMemoryBind, bufferMappingNum);
    Scratch<VkSparseBufferMemoryBindInfo> bufferInfos = NRI_ALLOCATE_SCRATCH(m_Device, VkSparseBufferMemoryBindInfo, bufferMappingNum);
    Scratch<VkSparseMemoryBind> opaqueBinds = NRI_ALLOCATE_SCRATCH(m_Device, VkSparseMemoryBind, opaqueCapacity);
    Scratch<VkSparseImageOpaqueMemoryBindInfo> opaqueInfos = NRI_ALLOCATE_SCRATCH(m_Device, VkSparseImageOpaqueMemoryBindInfo, textureMappingNum);
    Scratch<VkSparseImageMemoryBind> imageBinds = NRI_ALLOCATE_SCRATCH(m_Device, VkSparseImageMemoryBind, textureMappingNum * 3);
    Scratch<VkSparseImageMemoryBindInfo> imageInfos = NRI_ALLOCATE_SCRATCH(m_Device, VkSparseImageMemoryBindInfo, textureMappingNum);

    uint32_t bufferInfoNum = 0;
    uint32_t opaqueBindNum = 0;
    uint32_t opaqueInfoNum = 0;
    uint32_t imageInfoNum = 0;
    uint32_t imageBindNum = 0;

    for (uint32_t i = 0; i < bufferMappingNum; i++) {
        const SparseBufferMappingDesc& mapping = bufferMappings[i];
        const BufferVK& buffer = *(const BufferVK*)mapping.buffer;
        VkSparseMemoryBind& bind = bufferBinds[i];
        bind = {};
        bind.resourceOffset = mapping.offset;
        bind.size = mapping.size;

        if (mapping.mapping == SparseMapping::BIND) {
            const MemoryVK& memory = *(const MemoryVK*)mapping.memory;
            bind.memory = memory.GetHandle();
            bind.memoryOffset = memory.GetOffset() + mapping.memoryOffset;
        }

        const VkBuffer handle = buffer.GetHandle();

        if (bufferInfoNum == 0 || bufferInfos[bufferInfoNum - 1].buffer != handle) {
            VkSparseBufferMemoryBindInfo& info = bufferInfos[bufferInfoNum++];
            info = {};
            info.buffer = handle;
            info.pBinds = &bind;
        }

        bufferInfos[bufferInfoNum - 1].bindCount++;
    }

    for (uint32_t i = 0; i < textureMappingNum; i++) {
        const SparseTextureMappingDesc& mapping = textureMappings[i];
        const TextureVK& texture = *(const TextureVK*)mapping.texture;
        const TextureDesc& textureDesc = texture.GetDesc();
        const TextureTilingDesc& tiling = texture.GetTiling();
        const VkImage handle = texture.GetHandle();

        VkDeviceMemory memoryHandle = VK_NULL_HANDLE;
        VkDeviceSize memoryOffset = 0;

        if (mapping.mapping == SparseMapping::BIND) {
            const MemoryVK& memory = *(const MemoryVK*)mapping.memory;
            memoryHandle = memory.GetHandle();
            memoryOffset = memory.GetOffset() + mapping.memoryOffset;
        }

        if (mapping.metadata) {
            const uint32_t layers = texture.GetMetadataLayerNum();
            const VkDeviceSize layerSize = texture.GetMetadataSize();
            const uint32_t tileSize = m_Device.GetDesc().sparse.tileSize;

            if (layers == 0 || layerSize == 0 || tileSize == 0)
                return Result::FAILURE;

            const VkDeviceSize alignedLayerSize = ((layerSize + tileSize - 1) / tileSize) * tileSize;
            const uint32_t firstBind = opaqueBindNum;

            for (uint32_t layer = 0; layer < layers; layer++) {
                VkSparseMemoryBind& bind = opaqueBinds[opaqueBindNum];
                bind = {};
                bind.resourceOffset = texture.GetMetadataOffset() + (VkDeviceSize)layer * texture.GetMetadataStride();
                bind.size = layerSize;
                bind.memory = memoryHandle;
                bind.memoryOffset = memoryOffset + (VkDeviceSize)layer * alignedLayerSize;
                bind.flags = VK_SPARSE_MEMORY_BIND_METADATA_BIT;
                opaqueBindNum++;
            }

            if (opaqueInfoNum == 0 || opaqueInfos[opaqueInfoNum - 1].image != handle) {
                VkSparseImageOpaqueMemoryBindInfo& info = opaqueInfos[opaqueInfoNum];
                info = {};
                info.image = handle;
                info.pBinds = &opaqueBinds[firstBind];
                opaqueInfoNum++;
            }

            opaqueInfos[opaqueInfoNum - 1].bindCount += layers;

            continue;
        }

        if (IsSparsePackedTail(textureDesc, tiling, mapping.region)) {
            VkSparseMemoryBind& bind = opaqueBinds[opaqueBindNum];
            bind = {};
            bind.resourceOffset = texture.GetMipTailOffset();

            if (!tiling.singlePackedTail)
                bind.resourceOffset += (VkDeviceSize)mapping.region.layerOffset * texture.GetMipTailStride();

            bind.size = texture.GetMipTailSize();
            bind.memory = memoryHandle;
            bind.memoryOffset = memoryOffset;

            if (opaqueInfoNum == 0 || opaqueInfos[opaqueInfoNum - 1].image != handle) {
                VkSparseImageOpaqueMemoryBindInfo& info = opaqueInfos[opaqueInfoNum];
                info = {};
                info.image = handle;
                info.pBinds = &bind;
                opaqueInfoNum++;
            }

            opaqueInfos[opaqueInfoNum - 1].bindCount++;
            opaqueBindNum++;
        } else {
            const TextureRegionDesc expanded = ExpandWholeSizeRegion(GraphicsAPI::VK, textureDesc, mapping.region);
            const uint32_t firstBind = imageBindNum;

            VkSparseImageMemoryBind prototype = {};
            prototype.subresource.mipLevel = mapping.region.mipOffset;
            prototype.subresource.arrayLayer = mapping.region.layerOffset;
            prototype.offset = {(int32_t)expanded.x, (int32_t)expanded.y, (int32_t)expanded.z};
            prototype.extent = {expanded.width, expanded.height, expanded.depth};
            prototype.memory = memoryHandle;
            prototype.memoryOffset = memoryOffset;

            imageBindNum = AppendSparseAspects(texture.GetSparseAspectMask(), prototype, imageBinds, imageBindNum);

            if (imageBindNum == firstBind)
                return Result::FAILURE;

            if (imageInfoNum == 0 || imageInfos[imageInfoNum - 1].image != handle) {
                VkSparseImageMemoryBindInfo& info = imageInfos[imageInfoNum++];
                info = {};
                info.image = handle;
                info.pBinds = &imageBinds[firstBind];
            }

            imageInfos[imageInfoNum - 1].bindCount += imageBindNum - firstBind;
        }
    }

    VkBindSparseInfo bindInfo = {VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
    bindInfo.bufferBindCount = bufferInfoNum;
    bindInfo.pBufferBinds = bufferInfos;
    bindInfo.imageOpaqueBindCount = opaqueInfoNum;
    bindInfo.pImageOpaqueBinds = opaqueInfos;
    bindInfo.imageBindCount = imageInfoNum;
    bindInfo.pImageBinds = imageInfos;

    const auto& vk = m_Device.GetDispatchTable();
    const VkResult vkResult = vk.QueueBindSparse(m_Handle, 1, &bindInfo, VK_NULL_HANDLE);
    NRI_RETURN_ON_BAD_VKRESULT(&m_Device, vkResult, "vkQueueBindSparse");

    return Result::SUCCESS;
}

NRI_INLINE Result QueueVK::WaitIdle() {
    ExclusiveScope lock(*m_Lock);

    const auto& vk = m_Device.GetDispatchTable();
    VkResult vkResult = vk.QueueWaitIdle(m_Handle);
    NRI_RETURN_ON_BAD_VKRESULT(&m_Device, vkResult, "QueueWaitIdle");

    return Result::SUCCESS;
}

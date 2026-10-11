// © 2021 NVIDIA Corporation

TextureVK::~TextureVK() {
    if (m_OwnsNativeObjects) {
        const auto& vk = m_Device.GetDispatchTable();

        if (m_VmaAllocation)
            vmaDestroyImage(m_Device.GetVma(), m_Handle, m_VmaAllocation);
        else
            vk.DestroyImage(m_Device, m_Handle, m_Device.GetVkAllocationCallbacks());
    }
}

Result TextureVK::Create(const TextureDesc& textureDesc) {
    m_Desc = FixTextureDesc(textureDesc);

    if (m_Desc.usage & TextureUsageBits::SPARSE) {
        const DeviceDesc& deviceDesc = m_Device.GetDesc();

        if (!deviceDesc.features.sparseTexture || deviceDesc.sparse.tileSize == 0)
            return Result::UNSUPPORTED;

        if (m_Desc.type == TextureType::TEXTURE_1D)
            return Result::UNSUPPORTED;

        if (m_Desc.type == TextureType::TEXTURE_3D && deviceDesc.sparse.tier < 3)
            return Result::UNSUPPORTED;

        if (m_Desc.layerNum > 1 && deviceDesc.sparse.tier < 2)
            return Result::UNSUPPORTED;

        const Sample_t sampleNum = m_Desc.sampleNum;
        const bool supportedSampleCount = sampleNum == 1 || sampleNum == 2 || sampleNum == 4 || sampleNum == 8 || sampleNum == 16;

        if (!supportedSampleCount)
            return Result::UNSUPPORTED;

        if (sampleNum > 1) {
            const uint8_t sampleBit = (uint8_t)(sampleNum >> 1);

            if (deviceDesc.sparse.tier < 2 || (m_Device.GetSparseSampleMask() & sampleBit) == 0)
                return Result::UNSUPPORTED;
        }

        if (m_Desc.usage & (TextureUsageBits::VIDEO_DECODE | TextureUsageBits::VIDEO_ENCODE | TextureUsageBits::VIDEO_REFERENCE_ONLY))
            return Result::UNSUPPORTED;
    }

    VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    m_Device.FillCreateInfo(m_Desc, info);

    VideoResourceProfileListVK videoProfiles = {};
    const bool isVideoDecode = (m_Desc.usage & TextureUsageBits::VIDEO_DECODE) != 0;
    const bool isVideoEncode = (m_Desc.usage & TextureUsageBits::VIDEO_ENCODE) != 0;
    if (isVideoDecode || isVideoEncode) {
        videoProfiles.Fill(isVideoDecode, isVideoEncode, m_Desc.format, m_Desc.videoCodec, m_Device.GetVideoCodecOperations(isVideoDecode, isVideoEncode));
        if (videoProfiles.list.profileCount) {
            videoProfiles.list.pNext = info.pNext;
            info.pNext = &videoProfiles.list;
        }
    }

    const auto& vk = m_Device.GetDispatchTable();
    VkResult vkResult = vk.CreateImage(m_Device, &info, m_Device.GetVkAllocationCallbacks(), &m_Handle);
    NRI_RETURN_ON_BAD_VKRESULT(&m_Device, vkResult, "vkCreateImage");

    if (m_Desc.usage & TextureUsageBits::SPARSE)
        return CacheSparseRequirements();

    return Result::SUCCESS;
}

Result TextureVK::Create(const TextureVKDesc& textureVKDesc) {
    m_Desc = {};
    m_Desc.type = GetTextureType((VkImageType)textureVKDesc.vkImageType);
    m_Desc.format = VKFormatToNRIFormat((VkFormat)textureVKDesc.vkFormat);
    m_Desc.width = textureVKDesc.width;
    m_Desc.height = textureVKDesc.height;
    m_Desc.depth = textureVKDesc.depth;
    m_Desc.mipNum = textureVKDesc.mipNum;
    m_Desc.layerNum = textureVKDesc.layerNum;
    m_Desc.sampleNum = textureVKDesc.sampleNum;

    if (textureVKDesc.vkImageUsageFlags & VK_IMAGE_USAGE_SAMPLED_BIT)
        m_Desc.usage |= TextureUsageBits::SHADER_RESOURCE;

    if (textureVKDesc.vkImageUsageFlags & VK_IMAGE_USAGE_STORAGE_BIT)
        m_Desc.usage |= TextureUsageBits::SHADER_RESOURCE_STORAGE;

    if (textureVKDesc.vkImageUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        m_Desc.usage |= TextureUsageBits::COLOR_ATTACHMENT;

    if (textureVKDesc.vkImageUsageFlags & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
        m_Desc.usage |= TextureUsageBits::DEPTH_STENCIL_ATTACHMENT;

    if (textureVKDesc.vkImageUsageFlags & VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR)
        m_Desc.usage |= TextureUsageBits::SHADING_RATE_ATTACHMENT;

    if (textureVKDesc.vkImageUsageFlags & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)
        m_Desc.usage |= TextureUsageBits::INPUT_ATTACHMENT;

    const VkImageUsageFlags hostTransferUsage = m_Device.m_IsSupported.hostImageCopy
        ? VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT
        : VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    if ((textureVKDesc.vkImageUsageFlags & hostTransferUsage) == hostTransferUsage)
        m_Desc.usage |= TextureUsageBits::HOST_TRANSFER;

    if (textureVKDesc.vkImageUsageFlags & VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR)
        m_Desc.usage |= TextureUsageBits::VIDEO_DECODE;
    else if (textureVKDesc.vkImageUsageFlags & VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR)
        m_Desc.usage |= TextureUsageBits::VIDEO_DECODE | TextureUsageBits::VIDEO_REFERENCE_ONLY;

    if (textureVKDesc.vkImageUsageFlags & VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR)
        m_Desc.usage |= TextureUsageBits::VIDEO_ENCODE;
    else if (textureVKDesc.vkImageUsageFlags & VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR)
        m_Desc.usage |= TextureUsageBits::VIDEO_ENCODE | TextureUsageBits::VIDEO_REFERENCE_ONLY;

    m_OwnsNativeObjects = false;
    m_Handle = (VkImage)textureVKDesc.vkImage;

    return Result::SUCCESS;
}

Result TextureVK::AllocateAndBindMemory(MemoryLocation memoryLocation, float priority, bool committed) {
    if (m_Desc.usage & TextureUsageBits::SPARSE)
        return Result::UNSUPPORTED;

    NRI_CHECK(m_Handle, "Unexpected");

    MemoryDesc memoryDesc = {};
    GetMemoryDesc(memoryLocation, memoryDesc);

    MemoryTypeInfo memoryTypeInfo = Unpack(memoryDesc.type);
    if (memoryTypeInfo.mustBeDedicated)
        committed = true;

    VkMemoryRequirements memoryRequirements = {};
    memoryRequirements.size = memoryDesc.size;
    memoryRequirements.alignment = memoryDesc.alignment;
    memoryRequirements.memoryTypeBits = 1u << memoryTypeInfo.index;

    VmaAllocationCreateInfo allocationCreateInfo = {};
    allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT;
    allocationCreateInfo.flags |= committed ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT : VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT;
    allocationCreateInfo.flags |= IsHostVisibleMemory(memoryTypeInfo.location) ? VMA_ALLOCATION_CREATE_MAPPED_BIT : 0;
    allocationCreateInfo.priority = priority * 0.5f + 0.5f;
    allocationCreateInfo.memoryTypeBits = 1u << memoryTypeInfo.index; // "usage, requiredFlags and preferredFlags" not needed because of this

    VmaAllocationInfo allocationInfo = {};

    VkResult vkResult = vmaAllocateMemory(m_Device.GetVma(), &memoryRequirements, &allocationCreateInfo, &m_VmaAllocation, &allocationInfo);
    NRI_RETURN_ON_BAD_VKRESULT(&m_Device, vkResult, "vmaAllocateMemory");

    vkResult = vmaBindImageMemory(m_Device.GetVma(), m_VmaAllocation, m_Handle);
    NRI_RETURN_ON_BAD_VKRESULT(&m_Device, vkResult, "vmaBindImageMemory");

    return Result::SUCCESS;
}

Result TextureVK::BindMemory(const MemoryVK& memory, uint64_t offset) {
    if (m_Desc.usage & TextureUsageBits::SPARSE)
        return Result::UNSUPPORTED;

    NRI_CHECK(m_Handle, "Unexpected");
    NRI_CHECK(m_OwnsNativeObjects, "Not for wrapped objects");

    VkBindImageMemoryInfo bindImageMemoryInfo = {VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO};
    bindImageMemoryInfo.image = m_Handle;
    bindImageMemoryInfo.memory = memory.GetHandle();
    bindImageMemoryInfo.memoryOffset = memory.GetOffset() + offset;

    const auto& vk = m_Device.GetDispatchTable();
    VkResult vkResult = vk.BindImageMemory2(m_Device, 1, &bindImageMemoryInfo);
    NRI_RETURN_ON_BAD_VKRESULT(&m_Device, vkResult, "vkBindImageMemory2");

    return Result::SUCCESS;
}

void TextureVK::GetMemoryDesc(MemoryLocation memoryLocation, MemoryDesc& memoryDesc) const {
    VkMemoryDedicatedRequirements dedicatedRequirements = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};

    VkMemoryRequirements2 requirements = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    requirements.pNext = &dedicatedRequirements;

    VkImageMemoryRequirementsInfo2 imageMemoryRequirements = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2};
    imageMemoryRequirements.image = m_Handle;

    const auto& vk = m_Device.GetDispatchTable();
    vk.GetImageMemoryRequirements2(m_Device, &imageMemoryRequirements, &requirements);

    memoryDesc = {};
    m_Device.GetMemoryDesc(memoryLocation, requirements.memoryRequirements, dedicatedRequirements, memoryDesc);

    if (m_Desc.usage & TextureUsageBits::SPARSE) {
        const uint32_t tileSize = m_Device.GetDesc().sparse.tileSize;
        const uint64_t tileBytes = (uint64_t)m_Tiling.totalTileNum * tileSize;
        uint64_t bytes = tileBytes;

        if (m_Tiling.metadataSize > (UINT64_MAX - tileBytes))
            bytes = UINT64_MAX;
        else
            bytes += m_Tiling.metadataSize;

        if (tileSize != 0 && bytes != UINT64_MAX && (bytes % tileSize) != 0) {
            const uint64_t pad = (uint64_t)tileSize - (bytes % tileSize);

            if (pad > (UINT64_MAX - bytes))
                bytes = UINT64_MAX;
            else
                bytes += pad;
        }

        memoryDesc.size = bytes;

        if (tileSize != 0)
            memoryDesc.alignment = tileSize;

        ClearMustBeDedicated(memoryDesc);
    }
}

static bool MultiplySparseTiles(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > UINT64_MAX / a)
        return false;

    out = a * b;

    return true;
}

Result TextureVK::CacheSparseRequirements() {
    const auto& vk = m_Device.GetDispatchTable();
    const uint32_t tileSize = m_Device.GetDesc().sparse.tileSize;

    if (tileSize == 0)
        return Result::UNSUPPORTED;

    VkMemoryDedicatedRequirements dedicatedRequirements = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};

    VkMemoryRequirements2 requirements = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    requirements.pNext = &dedicatedRequirements;

    VkImageMemoryRequirementsInfo2 imageMemoryRequirements = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2};
    imageMemoryRequirements.image = m_Handle;

    vk.GetImageMemoryRequirements2(m_Device, &imageMemoryRequirements, &requirements);

    if (requirements.memoryRequirements.alignment != tileSize)
        return Result::UNSUPPORTED;

    uint32_t requirementNum = 0;
    vk.GetImageSparseMemoryRequirements(m_Device, m_Handle, &requirementNum, nullptr);

    if (requirementNum == 0)
        return Result::UNSUPPORTED;

    Scratch<VkSparseImageMemoryRequirements> sparseRequirements = NRI_ALLOCATE_SCRATCH(m_Device, VkSparseImageMemoryRequirements, requirementNum);

    for (uint32_t i = 0; i < requirementNum; i++)
        sparseRequirements[i] = {};

    vk.GetImageSparseMemoryRequirements(m_Device, m_Handle, &requirementNum, sparseRequirements);

    VkExtent3D granularity = {};
    VkSparseImageFormatFlags flags = 0;
    VkImageAspectFlags aspectMask = 0;
    VkDeviceSize mipTailSize = 0;
    VkDeviceSize mipTailOffset = 0;
    VkDeviceSize mipTailStride = 0;
    VkDeviceSize metadataOffset = 0;
    VkDeviceSize metadataSize = 0;
    VkDeviceSize metadataStride = 0;
    uint32_t metadataLayers = 0;
    uint32_t firstPackedMip = 0;
    bool hasStandard = false;
    bool hasMetadata = false;

    constexpr VkImageAspectFlags knownAspects = VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_METADATA_BIT;

    for (uint32_t i = 0; i < requirementNum; i++) {
        const VkSparseImageMemoryRequirements& requirement = sparseRequirements[i];
        const VkImageAspectFlags aspect = requirement.formatProperties.aspectMask;

        if ((aspect & ~knownAspects) != 0)
            return Result::UNSUPPORTED;

        if (aspect & VK_IMAGE_ASPECT_METADATA_BIT) {
            if (requirement.imageMipTailSize == 0)
                continue;

            if (hasMetadata)
                return Result::UNSUPPORTED;

            const bool singleTail = (requirement.formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) != 0;
            const uint32_t layers = singleTail ? 1u : m_Desc.layerNum;

            if (layers == 0)
                return Result::UNSUPPORTED;

            if (layers > 1 && requirement.imageMipTailStride < requirement.imageMipTailSize)
                return Result::UNSUPPORTED;

            hasMetadata = true;
            metadataOffset = requirement.imageMipTailOffset;
            metadataSize = requirement.imageMipTailSize;
            metadataStride = requirement.imageMipTailStride;
            metadataLayers = layers;

            continue;
        }

        if (aspect == 0)
            continue;

        if (!hasStandard) {
            hasStandard = true;
            granularity = requirement.formatProperties.imageGranularity;
            flags = requirement.formatProperties.flags;
            firstPackedMip = requirement.imageMipTailFirstLod;
            mipTailSize = requirement.imageMipTailSize;
            mipTailOffset = requirement.imageMipTailOffset;
            mipTailStride = requirement.imageMipTailStride;
        } else {
            const VkSparseImageFormatProperties& formatProperties = requirement.formatProperties;
            const bool sameGranularity = granularity.width == formatProperties.imageGranularity.width && granularity.height == formatProperties.imageGranularity.height && granularity.depth == formatProperties.imageGranularity.depth;
            const bool sameTail = flags == formatProperties.flags && firstPackedMip == requirement.imageMipTailFirstLod && mipTailSize == requirement.imageMipTailSize && mipTailOffset == requirement.imageMipTailOffset && mipTailStride == requirement.imageMipTailStride;

            if (!sameGranularity || !sameTail)
                return Result::UNSUPPORTED;
        }

        aspectMask |= aspect;
    }

    if (!hasStandard || granularity.width == 0 || granularity.height == 0 || granularity.depth == 0 || aspectMask == 0)
        return Result::UNSUPPORTED;

    // "imageGranularity" is in the same units as the mip extent. Report it as texels, matching D3D12 "WidthInTexels".
    const uint64_t tileWidth = granularity.width;
    const uint64_t tileHeight = granularity.height;
    const uint64_t tileDepth = granularity.depth;

    if (tileWidth == 0 || tileHeight == 0 || tileDepth == 0 || tileWidth > 0xFFFF || tileHeight > 0xFFFF || tileDepth > 0xFFFF)
        return Result::UNSUPPORTED;

    const bool hasTail = mipTailSize != 0;

    if (hasTail && firstPackedMip >= m_Desc.mipNum)
        return Result::UNSUPPORTED;

    const Dim_t packedMipNum = hasTail ? (Dim_t)(m_Desc.mipNum - firstPackedMip) : (Dim_t)0;
    const uint64_t packedTileNum = (mipTailSize + tileSize - 1) / tileSize;

    if (packedTileNum > 0xFFFFFFFFull)
        return Result::UNSUPPORTED;

    uint64_t totalTileNum = 0;
    const Dim_t standardMipNum = (Dim_t)(m_Desc.mipNum - packedMipNum);
    const bool is3D = m_Desc.type == TextureType::TEXTURE_3D;

    for (Dim_t mip = 0; mip < standardMipNum; mip++) {
        const uint32_t width = GetDimension(GraphicsAPI::VK, m_Desc, 0, mip);
        const uint32_t height = GetDimension(GraphicsAPI::VK, m_Desc, 1, mip);
        const uint32_t depth = GetDimension(GraphicsAPI::VK, m_Desc, 2, mip);
        const uint64_t tilesX = ((uint64_t)width + tileWidth - 1) / tileWidth;
        const uint64_t tilesY = ((uint64_t)height + tileHeight - 1) / tileHeight;
        const uint64_t tilesZ = ((uint64_t)depth + tileDepth - 1) / tileDepth;
        const uint64_t layers = is3D ? 1 : m_Desc.layerNum;
        uint64_t mipTiles = 0;

        if (!MultiplySparseTiles(tilesX, tilesY, mipTiles) || !MultiplySparseTiles(mipTiles, tilesZ, mipTiles) || !MultiplySparseTiles(mipTiles, layers, mipTiles))
            return Result::UNSUPPORTED;

        if (totalTileNum > UINT64_MAX - mipTiles)
            return Result::UNSUPPORTED;

        totalTileNum += mipTiles;
    }

    const bool singlePackedTail = hasTail && (flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) != 0;
    const uint64_t tailCopies = hasTail ? (singlePackedTail ? 1 : m_Desc.layerNum) : 0;
    uint64_t tailTiles = 0;

    if (!MultiplySparseTiles(packedTileNum, tailCopies, tailTiles) || totalTileNum > UINT64_MAX - tailTiles)
        return Result::UNSUPPORTED;

    totalTileNum += tailTiles;

    if (totalTileNum == 0 || totalTileNum > 0xFFFFFFFFull)
        return Result::UNSUPPORTED;

    uint64_t metadataBytes = 0;

    if (metadataLayers != 0) {
        if (metadataSize > (UINT64_MAX - (tileSize - 1)))
            return Result::UNSUPPORTED;

        const uint64_t alignedLayerSize = ((metadataSize + tileSize - 1) / tileSize) * tileSize;

        if (alignedLayerSize > (UINT64_MAX / metadataLayers))
            return Result::UNSUPPORTED;

        metadataBytes = alignedLayerSize * metadataLayers;
    }

    m_Tiling = {};
    m_Tiling.tileWidth = (Dim_t)tileWidth;
    m_Tiling.tileHeight = (Dim_t)tileHeight;
    m_Tiling.tileDepth = (Dim_t)tileDepth;
    m_Tiling.packedMipNum = packedMipNum;
    m_Tiling.packedTileNum = (uint32_t)packedTileNum;
    m_Tiling.totalTileNum = (uint32_t)totalTileNum;
    m_Tiling.metadataSize = metadataBytes;
    m_Tiling.singlePackedTail = singlePackedTail;
    m_MipTailOffset = mipTailOffset;
    m_MipTailSize = mipTailSize;
    m_MipTailStride = mipTailStride;
    m_MetadataOffset = metadataOffset;
    m_MetadataSize = metadataSize;
    m_MetadataStride = metadataStride;
    m_MetadataLayerNum = (uint16_t)metadataLayers;
    m_SparseAspectMask = aspectMask;

    return Result::SUCCESS;
}

NRI_INLINE void TextureVK::SetDebugName(const char* name) {
    m_Device.SetDebugNameToTrivialObject(VK_OBJECT_TYPE_IMAGE, (uint64_t)m_Handle, name);
}

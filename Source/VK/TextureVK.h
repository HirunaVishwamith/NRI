// © 2021 NVIDIA Corporation

#pragma once

namespace nri {

struct TextureVK final : public DebugNameBase {
    inline TextureVK(DeviceVK& device)
        : m_Device(device) {
    }

    inline VkImage GetHandle() const {
        return m_Handle;
    }

    inline DeviceVK& GetDevice() const {
        return m_Device;
    }

    inline VkExtent3D GetExtent() const {
        return {m_Desc.width, m_Desc.height, m_Desc.depth};
    }

    inline const TextureDesc& GetDesc() const {
        return m_Desc;
    }

    inline Dim_t GetSize(Dim_t dimensionIndex, Dim_t mip = 0) const {
        return GetDimension(GraphicsAPI::VK, m_Desc, dimensionIndex, mip);
    }

    ~TextureVK();

    Result Create(const TextureDesc& textureDesc);
    Result Create(const TextureVKDesc& textureVKDesc);
    Result AllocateAndBindMemory(MemoryLocation memoryLocation, float priority, bool committed);
    Result BindMemory(const MemoryVK& memory, uint64_t offset);
    inline const TextureTilingDesc& GetTiling() const {
        return m_Tiling;
    }

    inline VkDeviceSize GetMipTailOffset() const {
        return m_MipTailOffset;
    }

    inline VkDeviceSize GetMipTailSize() const {
        return m_MipTailSize;
    }

    inline VkDeviceSize GetMipTailStride() const {
        return m_MipTailStride;
    }

    inline VkImageAspectFlags GetSparseAspectMask() const {
        return m_SparseAspectMask;
    }

    inline VkDeviceSize GetMetadataOffset() const {
        return m_MetadataOffset;
    }

    inline VkDeviceSize GetMetadataSize() const {
        return m_MetadataSize;
    }

    inline VkDeviceSize GetMetadataStride() const {
        return m_MetadataStride;
    }

    inline uint32_t GetMetadataLayerNum() const {
        return m_MetadataLayerNum;
    }

    void GetMemoryDesc(MemoryLocation memoryLocation, MemoryDesc& memoryDesc) const;

    //================================================================================================================
    // DebugNameBase
    //================================================================================================================

    void SetDebugName(const char* name) NRI_DEBUG_NAME_OVERRIDE;

private:
    Result CacheSparseRequirements();

    DeviceVK& m_Device;
    VkImage m_Handle = VK_NULL_HANDLE;
    TextureDesc m_Desc = {};
    TextureTilingDesc m_Tiling = {};
    VkDeviceSize m_MipTailOffset = 0;
    VkDeviceSize m_MipTailSize = 0;
    VkDeviceSize m_MipTailStride = 0;
    VkDeviceSize m_MetadataOffset = 0;
    VkDeviceSize m_MetadataSize = 0;
    VkDeviceSize m_MetadataStride = 0;
    VkImageAspectFlags m_SparseAspectMask = 0;
    uint16_t m_MetadataLayerNum = 0;
    VmaAllocation m_VmaAllocation = nullptr;
    bool m_OwnsNativeObjects = true;
};

} // namespace nri

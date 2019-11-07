#pragma once

#include <vulkan\vulkan.h>
#include <vector>

class Model
{
	friend class Renderer;

private:
	VkBuffer m_vertex_buffer;
	VkDeviceMemory m_vertex_buffer_memory;
	std::vector<VkBuffer> m_uniform_buffers;
	std::vector<VkDeviceMemory> m_uniform_buffers_memory;
	std::vector<VkDescriptorSet> m_descriptor_sets;
	VkImage m_texture_image;
	VkDeviceMemory m_texture_image_memory;
	VkImageView m_texture_image_view;

	uint32_t m_num_vertices;
};

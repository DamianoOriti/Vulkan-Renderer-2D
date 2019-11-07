#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

#include <array>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <fstream>

#include "model.h"
#include "image.h"

class Renderer
{
public:
	struct Vertex;

	Renderer(GLFWwindow* window);
	Renderer(const Renderer&) = delete;
	~Renderer();

	void draw_frame(const glm::vec2& camera_position);
	void wait_idle();
	void signal_framebuffer_resize();
	void create_model(const std::vector<Vertex>& vertices, const char* image_path);

private:
	struct Queue_Family_Indices;
	struct Swap_Chain_Support_Details;
	struct Uniform_Buffer_Object;

	const std::vector<const char*> c_validation_layers = { "VK_LAYER_LUNARG_standard_validation" };
	const std::vector<const char*> c_device_extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

#ifdef NDEBUG
	const bool c_enable_validation_layers = false;
#else
	const bool c_enable_validation_layers = true;
#endif // NDEBUG

	const int c_max_frames_in_flight = 2;

	GLFWwindow* m_window;
	VkInstance m_instance;
	VkDebugUtilsMessengerEXT m_callback;
	VkSurfaceKHR m_surface;
	VkPhysicalDevice m_physical_device;
	VkDevice m_device;
	VkQueue m_graphics_queue;
	VkQueue m_present_queue;
	VkSwapchainKHR m_swap_chain;
	std::vector<VkImage> m_swap_chain_images;
	VkFormat m_swap_chain_image_format;
	VkExtent2D m_swap_chain_extent;
	std::vector<VkImageView> m_swap_chain_image_views;
	VkRenderPass m_render_pass;
	VkDescriptorSetLayout m_descriptor_set_layout;
	VkPipelineLayout m_pipeline_layout;
	VkPipeline m_graphics_pipeline;
	std::vector<VkFramebuffer> m_swap_chain_framebuffers;
	VkCommandPool m_command_pool;
	std::vector<VkCommandBuffer> m_command_buffers;
	std::vector<VkSemaphore> m_image_available_semaphores;
	std::vector<VkSemaphore> m_render_finished_semaphores;
	std::vector<VkFence> m_in_flight_fences;
	size_t m_current_frame;
	bool m_framebuffer_resized;
	VkDescriptorPool m_descriptor_pool;
	VkSampler m_texture_sampler;

	std::vector<Model> m_models;

	void create_instance();
	bool check_validation_layer_support();
	std::vector<const char*> get_required_extensions();
	void create_surface();

	static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
		VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
		VkDebugUtilsMessageTypeFlagsEXT message_type,
		const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
		void* user_data);

	static VkResult create_debug_utils_messenger_EXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* create_info,
		const VkAllocationCallbacks* allocator, VkDebugUtilsMessengerEXT* callback);

	static void destroy_debug_utils_messenger_ext(VkInstance instance, VkDebugUtilsMessengerEXT callback,
		const VkAllocationCallbacks* allocator);

	void setup_debug_callback();
	void pick_physical_device();
	bool is_device_suitable(VkPhysicalDevice physical_device);
	Queue_Family_Indices find_queue_families(VkPhysicalDevice device);
	bool check_device_extension_support(VkPhysicalDevice physical_device);
	void create_logical_device();
	Swap_Chain_Support_Details query_swap_chain_support(VkPhysicalDevice physical_device);
	VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats);
	VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR>& available_present_modes);
	VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities);
	void create_swap_chain();
	VkImageView create_image_view(VkImage image, VkFormat format);
	void create_graphics_pipeline();
	static std::vector<char> read_file(const std::string& filename);
	VkShaderModule create_shader_module(const std::vector<char>& code);
	void create_render_pass();
	void create_framebuffers();
	void create_command_pool();
	void create_command_buffers();
	void create_sync_objects();
	void clean_up_swap_chain();
	void recreate_swap_chain();
	void create_vertex_buffer(const std::vector<Vertex>& vertices, VkBuffer& vertex_buffer, VkDeviceMemory& vertex_buffer_memory);
	uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags property_flags);
	void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags memory_property_flags, VkBuffer& buffer, VkDeviceMemory& memory);
	void copy_buffer(VkBuffer& src_buffer, VkBuffer& dst_buffer, VkDeviceSize size);
	void create_descriptor_set_layout();
	void create_uniform_buffers(std::vector<VkBuffer>& uniform_buffers, std::vector<VkDeviceMemory>& uniform_buffers_memory);
	void create_descriptor_pool();
	void create_descriptor_sets(const std::vector<VkBuffer>& uniform_buffers, VkImageView texture_image_view, std::vector<VkDescriptorSet>& descriptor_sets);
	void create_texture_image(const char* image_path, VkImage& texture_image, VkDeviceMemory& texture_image_memory);
	void create_image(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memory_properties_flags, VkImage& image, VkDeviceMemory& image_memory);
	VkCommandBuffer begin_single_time_command();
	void end_single_time_command(VkCommandBuffer command_buffer);
	void transition_image_layout(VkImage image, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout);
	void copy_buffer_to_image(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
	void create_texture_sampler();

	void update_uniform_buffer(uint32_t image_index, Model& model, const glm::vec2& camera_position);
};

struct Renderer::Vertex
{
	glm::vec2 m_position;
	glm::vec3 m_color;
	glm::vec2 m_tex_coord;

	static VkVertexInputBindingDescription get_binding_description();
	static std::array<VkVertexInputAttributeDescription, 3> get_attribute_descriptions();
};

struct Renderer::Queue_Family_Indices
{
	int m_graphics_family = -1;
	int m_present_family = -1;

	bool is_complete();
};

struct Renderer::Swap_Chain_Support_Details
{
	VkSurfaceCapabilitiesKHR m_capabilities;
	std::vector<VkSurfaceFormatKHR> m_formats;
	std::vector<VkPresentModeKHR> m_present_modes;
};

struct Renderer::Uniform_Buffer_Object
{
	glm::vec2 m_camera_position;
};

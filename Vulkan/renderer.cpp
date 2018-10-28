#include "renderer.h"

Renderer::Renderer(GLFWwindow* window) :
	m_physical_device(VK_NULL_HANDLE),
	m_current_frame(0),
	m_window(window),
	m_framebuffer_resized(false)
{
	create_instance();
	setup_debug_callback();
	create_surface();
	pick_physical_device();
	create_logical_device();
	create_swap_chain();
	
	m_swap_chain_image_views.resize(m_swap_chain_images.size());
	for (size_t i = 0; i < m_swap_chain_images.size(); i++)
	{
		m_swap_chain_image_views[i] = create_image_view(m_swap_chain_images[i], m_swap_chain_image_format);
	}

	create_render_pass();
	create_descriptor_set_layout();
	create_graphics_pipeline();
	create_framebuffers();
	create_command_pool();
	create_descriptor_pool();
	create_command_buffers();
	create_sync_objects();
	create_texture_sampler();
}

Renderer::~Renderer()
{
	vkDestroySampler(m_device, m_texture_sampler, nullptr);

	clean_up_swap_chain();

	vkDestroyDescriptorPool(m_device, m_descriptor_pool, nullptr);

	vkDestroyDescriptorSetLayout(m_device, m_descriptor_set_layout, nullptr);

	for (auto model : m_models)
	{
		vkDestroyImageView(m_device, model.m_texture_image_view, nullptr);

		vkDestroyImage(m_device, model.m_texture_image, nullptr);
		vkFreeMemory(m_device, model.m_texture_image_memory, nullptr);

		for (size_t i = 0; i < model.m_uniform_buffers.size(); i++)
		{
			vkDestroyBuffer(m_device, model.m_uniform_buffers[i], nullptr);
			vkFreeMemory(m_device, model.m_uniform_buffers_memory[i], nullptr);
		}

		vkDestroyBuffer(m_device, model.m_vertex_buffer, nullptr);
		vkFreeMemory(m_device, model.m_vertex_buffer_memory, nullptr);
	}

	for (size_t i = 0; i < c_max_frames_in_flight; i++)
	{
		vkDestroyFence(m_device, m_in_flight_fences[i], nullptr);
		vkDestroySemaphore(m_device, m_render_finished_semaphores[i], nullptr);
		vkDestroySemaphore(m_device, m_image_available_semaphores[i], nullptr);
	}

	vkDestroyCommandPool(m_device, m_command_pool, nullptr);

	vkDestroyDevice(m_device, nullptr);

	if (enable_validation_layers)
	{
		destroy_debug_utils_messenger_ext(m_instance, m_callback, nullptr);
	}

	vkDestroySurfaceKHR(m_instance, m_surface, nullptr);

	vkDestroyInstance(m_instance, nullptr);
}

void Renderer::draw_frame(const glm::vec2& camera_position)
{
	vkWaitForFences(m_device, 1, &m_in_flight_fences[m_current_frame], VK_TRUE, std::numeric_limits<uint64_t>::max());

	uint32_t image_index;
	VkResult result = vkAcquireNextImageKHR(m_device, m_swap_chain, std::numeric_limits<uint64_t>::max(),
		m_image_available_semaphores[m_current_frame], VK_NULL_HANDLE, &image_index);

	if (result == VK_ERROR_OUT_OF_DATE_KHR)
	{
		recreate_swap_chain();
		return;
	}
	else if (result != VK_SUCCESS
		&& result != VK_SUBOPTIMAL_KHR)
	{
		throw std::runtime_error("failed to acquire swap chain image!");
	}


	// Update the uniform buffer for each model
	for (auto model : m_models)
	{
		update_uniform_buffer(image_index, model, camera_position);
	}


	// Submit draw command buffer
	VkSubmitInfo submit_info = {};
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	VkSemaphore wait_semaphores[] = { m_image_available_semaphores[m_current_frame] };
	VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submit_info.waitSemaphoreCount = 1;
	submit_info.pWaitSemaphores    = wait_semaphores;
	submit_info.pWaitDstStageMask  = wait_stages;

	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &m_command_buffers[image_index];

	VkSemaphore signal_semaphores[] = { m_render_finished_semaphores[m_current_frame] };
	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores    = signal_semaphores;

	vkResetFences(m_device, 1, &m_in_flight_fences[m_current_frame]);

	if (vkQueueSubmit(m_graphics_queue, 1, &submit_info, m_in_flight_fences[m_current_frame]) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to submit draw command buffer!");
	}


	// Present swap chain image
	VkPresentInfoKHR present_info = {};
	present_info.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores    = signal_semaphores;

	VkSwapchainKHR swap_chains[] = { m_swap_chain };
	present_info.swapchainCount = 1;
	present_info.pSwapchains    = swap_chains;
	present_info.pImageIndices  = &image_index;

	present_info.pResults = nullptr;	// Optional

	result = vkQueuePresentKHR(m_present_queue, &present_info);

	if (result == VK_ERROR_OUT_OF_DATE_KHR
		|| result == VK_SUBOPTIMAL_KHR
		|| m_framebuffer_resized)
	{
		m_framebuffer_resized = false;
		recreate_swap_chain();
	}
	else if (result != VK_SUCCESS)
	{
		throw std::runtime_error("failed to present swap chain image!");
	}


	// Increment current frame to avoid using the same frame for different draw commands simultaneously
	m_current_frame = (m_current_frame + 1) % c_max_frames_in_flight;
}

void Renderer::wait_idle()
{
	vkDeviceWaitIdle(m_device);
}

void Renderer::signal_framebuffer_resize()
{
	m_framebuffer_resized = true;
}

void Renderer::create_model(const std::vector<Vertex>& vertices, const char* image_path)
{
	Model model;

	model.m_num_vertices = static_cast<uint32_t>(vertices.size());

	create_vertex_buffer(vertices, model.m_vertex_buffer, model.m_vertex_buffer_memory);
	create_uniform_buffers(model.m_uniform_buffers, model.m_uniform_buffers_memory);

	create_texture_image(image_path, model.m_texture_image, model.m_texture_image_memory);

	model.m_texture_image_view = create_image_view(model.m_texture_image, VK_FORMAT_R8G8B8A8_UNORM);

	create_descriptor_sets(model.m_uniform_buffers, model.m_texture_image_view, model.m_descriptor_sets);


	m_models.push_back(model);

	// Update the command buffers to include the draw command for this model
	create_command_buffers();
}

void Renderer::create_instance()
{
	if (enable_validation_layers && !check_validation_layer_support())
	{
		throw std::runtime_error("validation layers requested, but not available!");
	}

	VkApplicationInfo app_info = {};
	app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app_info.pApplicationName   = "Vulkan Renderer";
	app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	app_info.pEngineName        = "VK Engine";
	app_info.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
	app_info.apiVersion         = VK_API_VERSION_1_0;

	VkInstanceCreateInfo create_info = {};
	create_info.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	create_info.pApplicationInfo = &app_info;

	auto extensions = get_required_extensions();
	create_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
	create_info.ppEnabledExtensionNames = extensions.data();

	if (enable_validation_layers)
	{
		create_info.enabledLayerCount   = static_cast<uint32_t>(c_validation_layers.size());
		create_info.ppEnabledLayerNames = c_validation_layers.data();
	}
	else
	{
		create_info.enabledLayerCount = 0;
	}

	VkResult result = vkCreateInstance(&create_info, nullptr, &m_instance);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("faild to create instance!");
	}
}

bool Renderer::check_validation_layer_support()
{
	uint32_t layer_count;
	vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

	std::vector<VkLayerProperties> available_layers(layer_count);
	vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

	for (const char* layer_name : c_validation_layers)
	{
		bool layer_found = false;

		for (const auto& layer_properties : available_layers)
		{
			if (strcmp(layer_name, layer_properties.layerName) == 0)
			{
				layer_found = true;
				break;
			}
		}

		if (!layer_found)
		{
			return false;
		}
	}

	return true;
}

std::vector<const char*> Renderer::get_required_extensions()
{
	uint32_t glfw_extension_count = 0;
	const char** glfw_extensions;
	glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

	std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

	if (enable_validation_layers)
	{
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
}

void Renderer::create_surface()
{
	if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create window surface!");
	}
}

VKAPI_ATTR VkBool32 VKAPI_CALL Renderer::debug_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
	VkDebugUtilsMessageTypeFlagsEXT message_type,
	const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
	void* user_data)
{
	std::cerr << "validation layer: " << callback_data->pMessage << std::endl;

	return VK_FALSE;
}

VkResult Renderer::create_debug_utils_messenger_EXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* create_info,
	const VkAllocationCallbacks* allocator, VkDebugUtilsMessengerEXT* callback)
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		return func(instance, create_info, allocator, callback);
	}
	else
	{
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void Renderer::destroy_debug_utils_messenger_ext(VkInstance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* allocator)
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		func(instance, callback, allocator);
	}
}

void Renderer::setup_debug_callback()
{
	if (!enable_validation_layers)
	{
		return;
	}

	VkDebugUtilsMessengerCreateInfoEXT create_info = {};
	create_info.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
		| VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
		| VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	create_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
		| VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
		| VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	create_info.pfnUserCallback = debug_callback;
	create_info.pUserData       = nullptr;

	if (create_debug_utils_messenger_EXT(m_instance, &create_info, nullptr, &m_callback) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to set up debug callback!");
	}
}

void Renderer::pick_physical_device()
{
	uint32_t device_count = 0;
	vkEnumeratePhysicalDevices(m_instance, &device_count, nullptr);

	if (device_count == 0)
	{
		throw std::runtime_error("failed to find GPUs with Vulkan support!");
	}

	std::vector<VkPhysicalDevice> devices(device_count);
	vkEnumeratePhysicalDevices(m_instance, &device_count, devices.data());

	for (const auto& device : devices)
	{
		if (is_device_suitable(device))
		{
			m_physical_device = device;
			break;
		}
	}

	if (m_physical_device == VK_NULL_HANDLE)
	{
		throw std::runtime_error("failed to find a suitable GPU!");
	}
}

bool Renderer::is_device_suitable(VkPhysicalDevice physical_device)
{
	Queue_Family_Indices indices = find_queue_families(physical_device);

	bool extension_supported = check_device_extension_support(physical_device);

	bool swap_chain_adequate = false;
	if (extension_supported)
	{
		Swap_Chain_Support_Details swap_chain_support_details = query_swap_chain_support(physical_device);
		swap_chain_adequate = !swap_chain_support_details.m_formats.empty()
			&& !swap_chain_support_details.m_present_modes.empty();
	}

	VkPhysicalDeviceFeatures physical_device_features;
	vkGetPhysicalDeviceFeatures(physical_device, &physical_device_features);

	return indices.is_complete()
		&& extension_supported
		&& swap_chain_adequate
		&& physical_device_features.samplerAnisotropy;
}

Renderer::Queue_Family_Indices Renderer::find_queue_families(VkPhysicalDevice device)
{
	Queue_Family_Indices indices;

	uint32_t queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

	std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

	int i = 0;
	for (const auto& queue_family : queue_families)
	{
		VkBool32 preset_support = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &preset_support);
		if (queue_family.queueCount > 0 && preset_support)
		{
			indices.m_present_family = i;
		}

		if (queue_family.queueCount > 0
			&& queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
		{
			indices.m_graphics_family = i;
		}

		if (indices.is_complete())
		{
			break;
		}
	}

	return indices;
}

bool Renderer::check_device_extension_support(VkPhysicalDevice physical_device)
{
	uint32_t extension_count;
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);

	std::vector<VkExtensionProperties> available_extensions(extension_count);
	vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, available_extensions.data());

	std::set<std::string> required_extensions(c_device_extensions.begin(), c_device_extensions.end());

	for (const auto& extension : available_extensions)
	{
		required_extensions.erase(extension.extensionName);
	}

	return required_extensions.empty();
}

void Renderer::create_logical_device()
{
	Queue_Family_Indices indices = find_queue_families(m_physical_device);

	std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
	std::set<int> unique_queue_families = { indices.m_graphics_family,indices.m_present_family };

	float queue_priority = 1.0f;
	for (int queue_family : unique_queue_families)
	{
		VkDeviceQueueCreateInfo queue_create_info = {};
		queue_create_info.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_create_info.queueFamilyIndex = queue_family;
		queue_create_info.queueCount       = 1;
		queue_create_info.pQueuePriorities = &queue_priority;

		queue_create_infos.push_back(queue_create_info);
	}

	VkPhysicalDeviceFeatures device_features = {};
	device_features.samplerAnisotropy = VK_TRUE;

	VkDeviceCreateInfo create_info = {};
	create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	create_info.pQueueCreateInfos       = queue_create_infos.data();
	create_info.queueCreateInfoCount    = static_cast<uint32_t>(queue_create_infos.size());
	create_info.pEnabledFeatures        = &device_features;
	create_info.enabledExtensionCount   = static_cast<uint32_t>(c_device_extensions.size());
	create_info.ppEnabledExtensionNames = c_device_extensions.data();

	if (enable_validation_layers)
	{
		create_info.enabledLayerCount   = static_cast<uint32_t>(c_validation_layers.size());
		create_info.ppEnabledLayerNames = c_validation_layers.data();
	}
	else
	{
		create_info.enabledLayerCount = 0;
	}

	if (vkCreateDevice(m_physical_device, &create_info, nullptr, &m_device) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create logical device!");
	}

	vkGetDeviceQueue(m_device, indices.m_graphics_family, 0, &m_graphics_queue);
	vkGetDeviceQueue(m_device, indices.m_present_family, 0, &m_present_queue);
}

Renderer::Swap_Chain_Support_Details Renderer::query_swap_chain_support(VkPhysicalDevice physical_device)
{
	Swap_Chain_Support_Details details;

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, m_surface, &details.m_capabilities);

	// get formats
	uint32_t format_count;
	vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, m_surface, &format_count, nullptr);

	if (format_count != 0)
	{
		details.m_formats.resize(format_count);
		vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, m_surface, &format_count, details.m_formats.data());
	}

	// get modes
	uint32_t present_mode_count;
	vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, m_surface, &present_mode_count, nullptr);

	if (present_mode_count != 0)
	{
		details.m_present_modes.resize(present_mode_count);
		vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, m_surface, &present_mode_count, details.m_present_modes.data());
	}

	return details;
}

VkSurfaceFormatKHR Renderer::choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& available_formats)
{
	if (available_formats.size() == 1
		&& available_formats[0].format == VK_FORMAT_UNDEFINED)
	{
		return{ VK_FORMAT_B8G8R8A8_UNORM,VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
	}

	for (const auto& available_format : available_formats)
	{
		if (available_format.format == VK_FORMAT_B8G8R8A8_UNORM
			&& available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
		{
			return available_format;
		}
	}

	return available_formats[0];
}

VkPresentModeKHR Renderer::choose_swap_present_mode(const std::vector<VkPresentModeKHR>& available_present_modes)
{
	VkPresentModeKHR best_mode = VK_PRESENT_MODE_FIFO_KHR;

	for (const auto& available_present_mode : available_present_modes)
	{
		if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR)
		{
			return available_present_mode;
		}
		else if (available_present_mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
		{
			best_mode = available_present_mode;
		}
	}

	return best_mode;
}

VkExtent2D Renderer::choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities)
{
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
	{
		return capabilities.currentExtent;
	}
	else
	{
		int width;
		int height;
		glfwGetFramebufferSize(m_window, &width, &height);

		VkExtent2D actual_extent = { width,height };
		actual_extent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actual_extent.width));
		actual_extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actual_extent.height));

		return actual_extent;
	}
}

void Renderer::create_swap_chain()
{
	Swap_Chain_Support_Details swap_chain_support_details = query_swap_chain_support(m_physical_device);

	VkSurfaceFormatKHR surface_format = choose_swap_surface_format(swap_chain_support_details.m_formats);
	VkPresentModeKHR present_mode = choose_swap_present_mode(swap_chain_support_details.m_present_modes);
	VkExtent2D extent = choose_swap_extent(swap_chain_support_details.m_capabilities);

	uint32_t image_count = swap_chain_support_details.m_capabilities.maxImageCount + 1;
	if (swap_chain_support_details.m_capabilities.maxImageCount > 0
		&& image_count > swap_chain_support_details.m_capabilities.maxImageCount)
	{
		image_count = swap_chain_support_details.m_capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR create_info = {};
	create_info.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	create_info.surface          = m_surface;
	create_info.minImageCount    = image_count;
	create_info.imageFormat      = surface_format.format;
	create_info.imageColorSpace  = surface_format.colorSpace;
	create_info.imageExtent      = extent;
	create_info.imageArrayLayers = 1;
	create_info.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	Queue_Family_Indices indices = find_queue_families(m_physical_device);
	uint32_t queue_family_indices[] = { (uint32_t)indices.m_graphics_family,(uint32_t)indices.m_present_family };

	if (indices.m_graphics_family!=indices.m_present_family)
	{
		create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
		create_info.queueFamilyIndexCount = 2;
		create_info.pQueueFamilyIndices   = queue_family_indices;
	}
	else
	{
		create_info.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
		create_info.queueFamilyIndexCount = 0;							// Optional
		create_info.pQueueFamilyIndices   = nullptr;					// Optional
	}

	create_info.preTransform   = swap_chain_support_details.m_capabilities.currentTransform;
	create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	create_info.presentMode    = present_mode;
	create_info.clipped        = VK_TRUE;
	create_info.oldSwapchain   = VK_NULL_HANDLE;

	if (vkCreateSwapchainKHR(m_device, &create_info, nullptr, &m_swap_chain) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create swap chain!");
	}

	vkGetSwapchainImagesKHR(m_device, m_swap_chain, &image_count, nullptr);
	m_swap_chain_images.resize(image_count);
	vkGetSwapchainImagesKHR(m_device, m_swap_chain, &image_count, m_swap_chain_images.data());

	m_swap_chain_image_format = surface_format.format;
	m_swap_chain_extent = extent;
}

VkImageView Renderer::create_image_view(VkImage image, VkFormat format)
{
	VkImageViewCreateInfo create_info = {};
	create_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	create_info.image                           = image;
	create_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
	create_info.format                          = format;
	create_info.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
	create_info.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
	create_info.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
	create_info.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
	create_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	create_info.subresourceRange.baseMipLevel   = 0;
	create_info.subresourceRange.levelCount     = 1;
	create_info.subresourceRange.baseArrayLayer = 0;
	create_info.subresourceRange.layerCount     = 1;

	VkImageView image_view;
	if (vkCreateImageView(m_device, &create_info, nullptr, &image_view) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create image views!");
	}

	return image_view;
}

void Renderer::create_graphics_pipeline()
{
	auto vert_shader_code = read_file("shaders/vert.spv");
	auto frag_shader_code = read_file("shaders/frag.spv");

	VkShaderModule vert_shader_module = create_shader_module(vert_shader_code);
	VkShaderModule frag_shader_module = create_shader_module(frag_shader_code);

	VkPipelineShaderStageCreateInfo vert_shader_stage_create_info = {};
	vert_shader_stage_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vert_shader_stage_create_info.stage  = VK_SHADER_STAGE_VERTEX_BIT;
	vert_shader_stage_create_info.module = vert_shader_module;
	vert_shader_stage_create_info.pName  = "main";

	VkPipelineShaderStageCreateInfo frag_shader_stage_create_info = {};
	frag_shader_stage_create_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	frag_shader_stage_create_info.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
	frag_shader_stage_create_info.module = frag_shader_module;
	frag_shader_stage_create_info.pName  = "main";

	VkPipelineShaderStageCreateInfo shader_stage_create_infos[] = { 
		vert_shader_stage_create_info,
		frag_shader_stage_create_info 
	};

	VkPipelineVertexInputStateCreateInfo vertex_input_state_create_info = {};
	vertex_input_state_create_info.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

	auto binding_description = Vertex::get_binding_description();
	auto attribute_descriptions = Vertex::get_attribute_descriptions();

	vertex_input_state_create_info.vertexBindingDescriptionCount   = 1;
	vertex_input_state_create_info.pVertexBindingDescriptions      = &binding_description;
	vertex_input_state_create_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_descriptions.size());
	vertex_input_state_create_info.pVertexAttributeDescriptions    = attribute_descriptions.data();

	VkPipelineInputAssemblyStateCreateInfo input_assembly_state_create_info = {};
	input_assembly_state_create_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	input_assembly_state_create_info.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	input_assembly_state_create_info.primitiveRestartEnable = VK_FALSE;

	VkViewport viewport = {};
	viewport.x        = 0.0f;
	viewport.y        = 0.0f;
	viewport.width    = static_cast<float>(m_swap_chain_extent.width);
	viewport.height   = static_cast<float>(m_swap_chain_extent.height);
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor = {};
	scissor.offset = { 0,0 };
	scissor.extent = m_swap_chain_extent;

	VkPipelineViewportStateCreateInfo viewport_state_create_info = {};
	viewport_state_create_info.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport_state_create_info.viewportCount = 1;
	viewport_state_create_info.pViewports    = &viewport;
	viewport_state_create_info.scissorCount  = 1;
	viewport_state_create_info.pScissors     = &scissor;

	VkPipelineRasterizationStateCreateInfo rasterization_state_create_info = {};
	rasterization_state_create_info.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterization_state_create_info.depthClampEnable        = VK_FALSE;
	rasterization_state_create_info.rasterizerDiscardEnable = VK_FALSE;
	rasterization_state_create_info.polygonMode             = VK_POLYGON_MODE_FILL;
	rasterization_state_create_info.lineWidth               = 1.0f;
	rasterization_state_create_info.cullMode                = VK_CULL_MODE_BACK_BIT;
	rasterization_state_create_info.frontFace               = VK_FRONT_FACE_CLOCKWISE;
	rasterization_state_create_info.depthBiasEnable         = VK_FALSE;
	rasterization_state_create_info.depthBiasConstantFactor = 0.0f;	// Optional
	rasterization_state_create_info.depthBiasClamp          = 0.0f;	// Optional
	rasterization_state_create_info.depthBiasSlopeFactor    = 0.0f;	// Optional

	VkPipelineMultisampleStateCreateInfo multisample_state_create_info = {};
	multisample_state_create_info.sType                 = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample_state_create_info.sampleShadingEnable   = VK_FALSE;
	multisample_state_create_info.rasterizationSamples  = VK_SAMPLE_COUNT_1_BIT;
	multisample_state_create_info.minSampleShading      = 1.0f;		// Optional
	multisample_state_create_info.pSampleMask           = nullptr;	// Optional
	multisample_state_create_info.alphaToCoverageEnable = VK_FALSE;	// Optional
	multisample_state_create_info.alphaToOneEnable      = VK_FALSE;	// Optional

	VkPipelineColorBlendAttachmentState color_blend_attachment_state = {};
	color_blend_attachment_state.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
		| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	color_blend_attachment_state.blendEnable         = VK_FALSE;
	color_blend_attachment_state.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;		// Optional
	color_blend_attachment_state.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;	// Optional
	color_blend_attachment_state.colorBlendOp        = VK_BLEND_OP_ADD;			// Optional
	color_blend_attachment_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;		// Optional
	color_blend_attachment_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;	// Optional
	color_blend_attachment_state.alphaBlendOp        = VK_BLEND_OP_ADD;			// Optional

	VkPipelineColorBlendStateCreateInfo color_blend_state_create_info = {};
	color_blend_state_create_info.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blend_state_create_info.logicOpEnable     = VK_FALSE;
	color_blend_state_create_info.logicOp           = VK_LOGIC_OP_COPY; // Optional
	color_blend_state_create_info.attachmentCount   = 1;
	color_blend_state_create_info.pAttachments      = &color_blend_attachment_state;
	color_blend_state_create_info.blendConstants[0] = 0.0f;	// Optional
	color_blend_state_create_info.blendConstants[1] = 0.0f; // Optional
	color_blend_state_create_info.blendConstants[2] = 0.0f; // Optional
	color_blend_state_create_info.blendConstants[3] = 0.0f; // Optional

	VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
	pipeline_layout_create_info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipeline_layout_create_info.setLayoutCount         = 1;
	pipeline_layout_create_info.pSetLayouts            = &m_descriptor_set_layout;
	pipeline_layout_create_info.pushConstantRangeCount = 0;			// Optional
	pipeline_layout_create_info.pPushConstantRanges    = nullptr;	// Optional

	if (vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &m_pipeline_layout) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create pipeline layout!");
	}

	VkGraphicsPipelineCreateInfo pipeline_create_info = {};
	pipeline_create_info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeline_create_info.stageCount          = 2;
	pipeline_create_info.pStages             = shader_stage_create_infos;
	pipeline_create_info.pVertexInputState   = &vertex_input_state_create_info;
	pipeline_create_info.pInputAssemblyState = &input_assembly_state_create_info;
	pipeline_create_info.pViewportState      = &viewport_state_create_info;
	pipeline_create_info.pRasterizationState = &rasterization_state_create_info;
	pipeline_create_info.pMultisampleState   = &multisample_state_create_info;
	pipeline_create_info.pDepthStencilState  = nullptr;								// Optional
	pipeline_create_info.pColorBlendState    = &color_blend_state_create_info;
	pipeline_create_info.pDynamicState       = nullptr;								// Optional
	pipeline_create_info.layout              = m_pipeline_layout;
	pipeline_create_info.renderPass          = m_render_pass;
	pipeline_create_info.subpass             = 0;
	pipeline_create_info.basePipelineHandle  = VK_NULL_HANDLE;
	pipeline_create_info.basePipelineIndex   = -1;									// Optional

	if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipeline_create_info, nullptr, &m_graphics_pipeline) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create graphics pipeline!");
	}

	vkDestroyShaderModule(m_device, frag_shader_module, nullptr);
	vkDestroyShaderModule(m_device, vert_shader_module, nullptr);
}

std::vector<char> Renderer::read_file(const std::string& filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open())
	{
		throw std::runtime_error("failed to open file!");
	}

	size_t file_size = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(file_size);

	file.seekg(0);
	file.read(buffer.data(), file_size);

	file.close();

	return buffer;
}

VkShaderModule Renderer::create_shader_module(const std::vector<char>& code)
{
	VkShaderModuleCreateInfo create_info = {};
	create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	create_info.codeSize = code.size();
	create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

	VkShaderModule shader_module;
	if (vkCreateShaderModule(m_device,&create_info,nullptr,&shader_module)!=VK_SUCCESS)
	{
		throw std::runtime_error("failed to create shader module!");
	}

	return shader_module;
}

void Renderer::create_render_pass()
{
	VkAttachmentDescription color_attachment_desc = {};
	color_attachment_desc.format         = m_swap_chain_image_format;
	color_attachment_desc.samples        = VK_SAMPLE_COUNT_1_BIT;
	color_attachment_desc.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
	color_attachment_desc.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment_desc.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment_desc.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment_desc.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference color_attachment_ref = {};
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	
	VkSubpassDescription subpass_desc = {};
	subpass_desc.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass_desc.colorAttachmentCount = 1;
	subpass_desc.pColorAttachments    = &color_attachment_ref;

	VkRenderPassCreateInfo render_pass_create_info = {};
	render_pass_create_info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_create_info.attachmentCount = 1;
	render_pass_create_info.pAttachments    = &color_attachment_desc;
	render_pass_create_info.subpassCount    = 1;
	render_pass_create_info.pSubpasses      = &subpass_desc;

	VkSubpassDependency subpass_dependency = {};
	subpass_dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
	subpass_dependency.dstSubpass    = 0;
	subpass_dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	subpass_dependency.srcAccessMask = 0;
	subpass_dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	subpass_dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	render_pass_create_info.dependencyCount = 1;
	render_pass_create_info.pDependencies   = &subpass_dependency;

	if (vkCreateRenderPass(m_device, &render_pass_create_info, nullptr, &m_render_pass) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create render pass!");
	}
}

void Renderer::create_framebuffers()
{
	m_swap_chain_framebuffers.resize(m_swap_chain_image_views.size());

	for (size_t i = 0; i < m_swap_chain_image_views.size(); i++)
	{
		VkImageView attachments[] = { m_swap_chain_image_views[i] };

		VkFramebufferCreateInfo framebuffer_create_info = {};
		framebuffer_create_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebuffer_create_info.renderPass      = m_render_pass;
		framebuffer_create_info.attachmentCount = 1;
		framebuffer_create_info.pAttachments    = attachments;
		framebuffer_create_info.width           = m_swap_chain_extent.width;
		framebuffer_create_info.height          = m_swap_chain_extent.height;
		framebuffer_create_info.layers          = 1;

		if (vkCreateFramebuffer(m_device, &framebuffer_create_info, nullptr, &m_swap_chain_framebuffers[i]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create framebuffer!");
		}
	}
}

void Renderer::create_command_pool()
{
	Queue_Family_Indices queue_family_indices = find_queue_families(m_physical_device);

	VkCommandPoolCreateInfo create_info = {};
	create_info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	create_info.queueFamilyIndex = queue_family_indices.m_graphics_family;
	create_info.flags            = 0;	// Optional

	if (vkCreateCommandPool(m_device, &create_info, nullptr, &m_command_pool) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create command pool!");
	}
}

void Renderer::create_command_buffers()
{
	m_command_buffers.resize(m_swap_chain_framebuffers.size());

	VkCommandBufferAllocateInfo alloc_info = {};
	alloc_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	alloc_info.commandPool        = m_command_pool;
	alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	alloc_info.commandBufferCount = static_cast<uint32_t>(m_command_buffers.size());
	
	if (vkAllocateCommandBuffers(m_device, &alloc_info, m_command_buffers.data()) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate command buffers!");
	}

	for (size_t i = 0; i < m_command_buffers.size(); i++)
	{
		VkCommandBufferBeginInfo command_buffer_begin_info = {};
		command_buffer_begin_info.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		command_buffer_begin_info.flags            = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		command_buffer_begin_info.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(m_command_buffers[i], &command_buffer_begin_info) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkRenderPassBeginInfo render_pass_begin_info = {};
		render_pass_begin_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		render_pass_begin_info.renderPass        = m_render_pass;
		render_pass_begin_info.framebuffer       = m_swap_chain_framebuffers[i];
		render_pass_begin_info.renderArea.offset = {0, 0};
		render_pass_begin_info.renderArea.extent = m_swap_chain_extent;
		
		VkClearValue clear_value = {0.0f, 0.0f, 0.0f, 1.0f};
		render_pass_begin_info.clearValueCount = 1;
		render_pass_begin_info.pClearValues    = &clear_value;

		vkCmdBeginRenderPass(m_command_buffers[i], &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(m_command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphics_pipeline);

		for (Model model : m_models)
		{
			VkBuffer vertex_buffers[] = { model.m_vertex_buffer };
			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(m_command_buffers[i], 0, 1, vertex_buffers, offsets);
			
			vkCmdBindDescriptorSets(m_command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline_layout, 0, 1, &model.m_descriptor_sets[i], 0, nullptr);

			vkCmdDraw(m_command_buffers[i], model.m_num_vertices, 1, 0, 0);
		}

		vkCmdEndRenderPass(m_command_buffers[i]);

		if (vkEndCommandBuffer(m_command_buffers[i]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to record command buffer!");
		}
	}
}

void Renderer::create_sync_objects()
{
	m_image_available_semaphores.resize(c_max_frames_in_flight);
	m_render_finished_semaphores.resize(c_max_frames_in_flight);
	m_in_flight_fences.resize(c_max_frames_in_flight);

	VkSemaphoreCreateInfo semaphore_create_info = {};
	semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fence_create_info = {};
	fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (size_t i = 0; i < c_max_frames_in_flight; i++)
	{
		if (vkCreateSemaphore(m_device, &semaphore_create_info, nullptr, &m_image_available_semaphores[i]) != VK_SUCCESS
			|| vkCreateSemaphore(m_device, &semaphore_create_info, nullptr, &m_render_finished_semaphores[i]) != VK_SUCCESS
			|| vkCreateFence(m_device, &fence_create_info, nullptr, &m_in_flight_fences[i]) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create synchronization objects for a frame!");
		}
	}
}

void Renderer::clean_up_swap_chain()
{
	for (auto framebuffer : m_swap_chain_framebuffers)
	{
		vkDestroyFramebuffer(m_device, framebuffer, nullptr);
	}

	vkFreeCommandBuffers(m_device, m_command_pool, static_cast<uint32_t>(m_command_buffers.size()), m_command_buffers.data());

	vkDestroyPipeline(m_device, m_graphics_pipeline, nullptr);

	vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);

	vkDestroyRenderPass(m_device, m_render_pass, nullptr);

	for (auto image_view : m_swap_chain_image_views)
	{
		vkDestroyImageView(m_device, image_view, nullptr);
	}

	vkDestroySwapchainKHR(m_device, m_swap_chain, nullptr);
}

void Renderer::recreate_swap_chain()
{
	int width  = 0;
	int height = 0;
	while (width == 0 || height == 0)
	{
		glfwGetFramebufferSize(m_window, &width, &height);
		glfwWaitEvents();
	}

	vkDeviceWaitIdle(m_device);

	clean_up_swap_chain();

	create_swap_chain();

	m_swap_chain_image_views.resize(m_swap_chain_images.size());
	for (size_t i = 0; i < m_swap_chain_images.size(); i++)
	{
		m_swap_chain_image_views[i] = create_image_view(m_swap_chain_images[i], m_swap_chain_image_format);
	}

	create_render_pass();
	create_graphics_pipeline();
	create_framebuffers();
	create_command_buffers();
}

void Renderer::create_vertex_buffer(const std::vector<Vertex>& vertices, VkBuffer& vertex_buffer, VkDeviceMemory& vertex_buffer_memory)
{
	VkDeviceSize buffer_size = sizeof(vertices[0]) * vertices.size();


	// Create the staging buffer
	VkBuffer staging_buffer;
	VkDeviceMemory staging_buffer_memory;
	create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);


	// Fill the staging buffer with the vertex data
	void* data;
	vkMapMemory(m_device, staging_buffer_memory, 0, buffer_size, 0, &data);
	memcpy(data, vertices.data(), static_cast<size_t>(buffer_size));
	vkUnmapMemory(m_device, staging_buffer_memory);


	// Create the buffer which will actually be used
	create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertex_buffer, vertex_buffer_memory);


	// Copy the data from the staging buffer to the final buffer
	copy_buffer(staging_buffer, vertex_buffer, buffer_size);


	// Clean up the staging buffer and the memory allocated for it
	vkDestroyBuffer(m_device, staging_buffer, nullptr);
	vkFreeMemory(m_device, staging_buffer_memory, nullptr);
}

uint32_t Renderer::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags property_flags)
{
	VkPhysicalDeviceMemoryProperties memory_properties;
	vkGetPhysicalDeviceMemoryProperties(m_physical_device, &memory_properties);

	for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
	{
		if ((type_filter & (1 << i))
			&& (memory_properties.memoryTypes[i].propertyFlags & property_flags) == property_flags)
		{
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

void Renderer::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags memory_property_flags, VkBuffer& buffer, VkDeviceMemory& memory)
{
	VkBufferCreateInfo buffer_create_info = {};
	buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_create_info.size  = size;
	buffer_create_info.usage = usage_flags;
	buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	if (vkCreateBuffer(m_device, &buffer_create_info, nullptr, &buffer) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create vertex buffer!");
	}

	// Memory allocation
	VkMemoryRequirements memory_requirements;
	vkGetBufferMemoryRequirements(m_device, buffer, &memory_requirements);

	VkMemoryAllocateInfo memory_allocate_info = {};
	memory_allocate_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memory_allocate_info.allocationSize  = memory_requirements.size;
	memory_allocate_info.memoryTypeIndex = find_memory_type(memory_requirements.memoryTypeBits, memory_property_flags);

	if (vkAllocateMemory(m_device, &memory_allocate_info, nullptr, &memory) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate vertex buffer memory!");
	}

	vkBindBufferMemory(m_device, buffer, memory, 0);
}

void Renderer::copy_buffer(VkBuffer& src_buffer, VkBuffer& dst_buffer, VkDeviceSize size)
{
	VkCommandBuffer command_buffer = begin_single_time_command();


	// Record the copy command
	VkBufferCopy copy_region = {};
	copy_region.srcOffset = 0;		// Optional
	copy_region.dstOffset = 0;		// Optional
	copy_region.size      = size;

	vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);


	end_single_time_command(command_buffer);
}

void Renderer::create_descriptor_set_layout()
{
	VkDescriptorSetLayoutBinding ubo_layout_binding = {};
	ubo_layout_binding.binding            = 0;
	ubo_layout_binding.descriptorType     = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	ubo_layout_binding.descriptorCount    = 1;
	ubo_layout_binding.stageFlags         = VK_SHADER_STAGE_VERTEX_BIT;
	ubo_layout_binding.pImmutableSamplers = nullptr;	// Optional

	VkDescriptorSetLayoutBinding texture_sampler_layout_binding = {};
	texture_sampler_layout_binding.binding            = 1;
	texture_sampler_layout_binding.descriptorCount    = 1;
	texture_sampler_layout_binding.descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	texture_sampler_layout_binding.pImmutableSamplers = nullptr;
	texture_sampler_layout_binding.stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT;

	std::array<VkDescriptorSetLayoutBinding, 2> bindings = { ubo_layout_binding,texture_sampler_layout_binding };

	VkDescriptorSetLayoutCreateInfo layout_create_info = {};
	layout_create_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layout_create_info.bindingCount = static_cast<uint32_t>(bindings.size());
	layout_create_info.pBindings    = bindings.data();

	if (vkCreateDescriptorSetLayout(m_device, &layout_create_info, nullptr, &m_descriptor_set_layout) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create descriptor set layout!");
	}
}

void Renderer::create_uniform_buffers(std::vector<VkBuffer>& uniform_buffers, std::vector<VkDeviceMemory>& uniform_buffers_memory)
{
	VkDeviceSize buffer_size = sizeof(Uniform_Buffer_Object);

	uniform_buffers.resize(m_swap_chain_images.size());
	uniform_buffers_memory.resize(m_swap_chain_images.size());

	for (size_t i = 0; i < m_swap_chain_images.size(); i++)
	{
		VkBufferUsageFlags usage_flags = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		VkMemoryPropertyFlags memory_property_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		create_buffer(buffer_size, usage_flags, memory_property_flags, uniform_buffers[i], uniform_buffers_memory[i]);
	}
}

void Renderer::create_descriptor_pool()
{
	std::array<VkDescriptorPoolSize, 2> pool_sizes = {};
	pool_sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	pool_sizes[0].descriptorCount = static_cast<uint32_t>(m_swap_chain_images.size()) * 64;
	pool_sizes[1].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	pool_sizes[1].descriptorCount = static_cast<uint32_t>(m_swap_chain_images.size()) * 64;

	VkDescriptorPoolCreateInfo pool_create_info = {};
	pool_create_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_create_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
	pool_create_info.pPoolSizes    = pool_sizes.data();
	pool_create_info.maxSets       = static_cast<uint32_t>(m_swap_chain_images.size()) * 64;
	
	if (vkCreateDescriptorPool(m_device, &pool_create_info, nullptr, &m_descriptor_pool) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create descriptor pool!");
	}
}

void Renderer::create_descriptor_sets(const std::vector<VkBuffer>& uniform_buffers, VkImageView texture_image_view, std::vector<VkDescriptorSet>& descriptor_sets)
{
	// Allocate the descriptor sets
	std::vector<VkDescriptorSetLayout> layouts(m_swap_chain_images.size(), m_descriptor_set_layout);

	VkDescriptorSetAllocateInfo set_allocate_info = {};
	set_allocate_info.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	set_allocate_info.descriptorPool     = m_descriptor_pool;
	set_allocate_info.descriptorSetCount = static_cast<uint32_t>(m_swap_chain_images.size());
	set_allocate_info.pSetLayouts        = layouts.data();

	descriptor_sets.resize(m_swap_chain_images.size());
	if (vkAllocateDescriptorSets(m_device, &set_allocate_info, descriptor_sets.data()) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate descriptor sets!");
	}


	// Configure the descriptor sets
	for (size_t i = 0; i < m_swap_chain_images.size(); i++)
	{
		VkDescriptorBufferInfo ubo_info = {};
		ubo_info.buffer = uniform_buffers[i];
		ubo_info.offset = 0;
		ubo_info.range  = sizeof(Uniform_Buffer_Object);

		VkDescriptorImageInfo image_info = {};
		image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		image_info.imageView   = texture_image_view;
		image_info.sampler     = m_texture_sampler;

		std::array<VkWriteDescriptorSet, 2> descriptor_set_writes = {};
		descriptor_set_writes[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptor_set_writes[0].dstSet           = descriptor_sets[i];
		descriptor_set_writes[0].dstBinding       = 0;
		descriptor_set_writes[0].dstArrayElement  = 0;
		descriptor_set_writes[0].descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptor_set_writes[0].descriptorCount  = 1;
		descriptor_set_writes[0].pBufferInfo      = &ubo_info;

		descriptor_set_writes[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptor_set_writes[1].dstSet           = descriptor_sets[i];
		descriptor_set_writes[1].dstBinding       = 1;
		descriptor_set_writes[1].dstArrayElement  = 0;
		descriptor_set_writes[1].descriptorType   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptor_set_writes[1].descriptorCount  = 1;
		descriptor_set_writes[1].pImageInfo       = &image_info;

		vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(descriptor_set_writes.size()), descriptor_set_writes.data(), 0, nullptr);
	}
}

void Renderer::create_texture_image(const char* image_path, VkImage& texture_image, VkDeviceMemory& texture_image_memory)
{
	Image image(image_path);


	// Create the staging buffer and copy the image data to it 
	VkBuffer staging_buffer;
	VkDeviceMemory staging_buffer_memory;

	VkDeviceSize image_size = image.get_width() * image.get_height() * 4;
	VkMemoryPropertyFlags memory_property_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	create_buffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, memory_property_flags, staging_buffer, staging_buffer_memory);

	void* data;
	vkMapMemory(m_device, staging_buffer_memory, 0, image_size, 0, &data);
	memcpy(data, image.get_data(), static_cast<size_t>(image_size));
	vkUnmapMemory(m_device, staging_buffer_memory);


	// Create the texture image
	create_image(image.get_width(), image.get_height(), VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, texture_image, texture_image_memory);


	// Transition the texture image to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
	transition_image_layout(texture_image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);


	copy_buffer_to_image(staging_buffer, texture_image, static_cast<uint32_t>(image.get_width()), static_cast<uint32_t>(image.get_height()));


	// Transition the texture image to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL for sampling it in the shader
	transition_image_layout(texture_image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);


	// Clean up
	vkDestroyBuffer(m_device, staging_buffer, nullptr);
	vkFreeMemory(m_device, staging_buffer_memory, nullptr);
}

void Renderer::create_image(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memory_properties_flags, VkImage& image, VkDeviceMemory& image_memory)
{
	// Create the image
	VkImageCreateInfo create_info = {};
	create_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	create_info.imageType     = VK_IMAGE_TYPE_2D;
	create_info.extent.width  = width;
	create_info.extent.height = height;
	create_info.extent.depth  = 1;
	create_info.mipLevels     = 1;
	create_info.arrayLayers   = 1;
	create_info.format        = format;
	create_info.tiling        = tiling;
	create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	create_info.usage         = usage;
	create_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
	create_info.samples       = VK_SAMPLE_COUNT_1_BIT;
	create_info.flags         = 0;	// Optional

	if (vkCreateImage(m_device, &create_info, nullptr, &image) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create image!");
	}


	// Allocate memory for the image
	VkMemoryRequirements memory_requirements;
	vkGetImageMemoryRequirements(m_device, image, &memory_requirements);

	VkMemoryAllocateInfo memory_allocate_info = {};
	memory_allocate_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memory_allocate_info.allocationSize  = memory_requirements.size;
	memory_allocate_info.memoryTypeIndex = find_memory_type(memory_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	if (vkAllocateMemory(m_device, &memory_allocate_info, nullptr, &image_memory) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate memory for image!");
	}


	// Finally bind the allocated memory to the image
	vkBindImageMemory(m_device, image, image_memory, 0);
}

VkCommandBuffer Renderer::begin_single_time_command()
{
	// Allocate temporary command buffer
	VkCommandBufferAllocateInfo command_buffer_allocate_info = {};
	command_buffer_allocate_info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	command_buffer_allocate_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	command_buffer_allocate_info.commandPool        = m_command_pool;
	command_buffer_allocate_info.commandBufferCount = 1;

	VkCommandBuffer command_buffer;
	if (vkAllocateCommandBuffers(m_device, &command_buffer_allocate_info, &command_buffer) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to allocate command buffer!");
	}


	// Begin recording command buffer
	VkCommandBufferBeginInfo command_buffer_begin_info = {};
	command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	if (vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to record command buffer!");
	}

	return command_buffer;
}

void Renderer::end_single_time_command(VkCommandBuffer command_buffer)
{
	vkEndCommandBuffer(command_buffer);


	// Execute the command buffer
	VkSubmitInfo submit_info = {};
	submit_info.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &command_buffer;

	vkQueueSubmit(m_graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
	vkQueueWaitIdle(m_graphics_queue);


	// Clean up the command buffer
	vkFreeCommandBuffers(m_device, m_command_pool, 1, &command_buffer);
}

void Renderer::transition_image_layout(VkImage image, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout)
{
	VkCommandBuffer command_buffer = begin_single_time_command();

	VkImageMemoryBarrier barrier = {};
	barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout                       = old_layout;
	barrier.newLayout                       = new_layout;
	barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                           = image;
	barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel   = 0;
	barrier.subresourceRange.levelCount     = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount     = 1;

	VkPipelineStageFlags src_stage_flags;
	VkPipelineStageFlags dst_stage_flags;

	if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED
		&& new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		src_stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dst_stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}
	else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
		&& new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		src_stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
		dst_stage_flags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	}
	else
	{
		throw std::runtime_error("unsupported layout transition!");
	}

	vkCmdPipelineBarrier(command_buffer, src_stage_flags, dst_stage_flags, 0, 0, nullptr, 0, nullptr, 1, &barrier);

	end_single_time_command(command_buffer);
}

void Renderer::copy_buffer_to_image(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
	VkCommandBuffer command_buffer = begin_single_time_command();

	VkBufferImageCopy region = {};
	region.bufferOffset      = 0;
	region.bufferRowLength   = 0;
	region.bufferImageHeight = 0;

	region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel       = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount     = 1;
	
	region.imageOffset = {0, 0, 0};
	region.imageExtent = {width, height, 1};

	vkCmdCopyBufferToImage(command_buffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	end_single_time_command(command_buffer);
}

void Renderer::create_texture_sampler()
{
	VkSamplerCreateInfo create_info = {};
	create_info.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	create_info.magFilter               = VK_FILTER_NEAREST;
	create_info.minFilter               = VK_FILTER_NEAREST;
	create_info.addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	create_info.addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	create_info.addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	create_info.anisotropyEnable        = VK_TRUE;
	create_info.maxAnisotropy           = 16;
	create_info.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	create_info.unnormalizedCoordinates = VK_FALSE;
	create_info.compareEnable           = VK_FALSE;
	create_info.compareOp               = VK_COMPARE_OP_ALWAYS;
	create_info.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	create_info.mipLodBias              = 0.0f;
	create_info.minLod                  = 0.0f;
	create_info.maxLod                  = 0.0f;

	if (vkCreateSampler(m_device, &create_info, nullptr, &m_texture_sampler) != VK_SUCCESS)
	{
		throw std::runtime_error("failed to create texture sampler!");
	}
}

void Renderer::update_uniform_buffer(uint32_t image_index, Model& model, const glm::vec2& camera_position)
{
	Uniform_Buffer_Object ubo;
	ubo.m_camera_position = camera_position;

	void* data;
	vkMapMemory(m_device, model.m_uniform_buffers_memory[image_index], 0, sizeof(ubo), 0, &data);
	memcpy(data, &ubo, sizeof(ubo));
	vkUnmapMemory(m_device, model.m_uniform_buffers_memory[image_index]);
}

bool Renderer::Queue_Family_Indices::is_complete()
{
	return m_graphics_family >= 0 && m_present_family >= 0;
}

VkVertexInputBindingDescription Renderer::Vertex::get_binding_description()
{
	VkVertexInputBindingDescription binding_description = {};
	binding_description.binding   = 0;
	binding_description.stride    = sizeof(Vertex);
	binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	return binding_description;
}

std::array<VkVertexInputAttributeDescription, 3> Renderer::Vertex::get_attribute_descriptions()
{
	std::array<VkVertexInputAttributeDescription, 3> attribute_descriptions = {};
	attribute_descriptions[0].binding  = 0;
	attribute_descriptions[0].location = 0;
	attribute_descriptions[0].format   = VK_FORMAT_R32G32_SFLOAT;
	attribute_descriptions[0].offset   = offsetof(Vertex, m_position);

	attribute_descriptions[1].binding  = 0;
	attribute_descriptions[1].location = 1;
	attribute_descriptions[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
	attribute_descriptions[1].offset   = offsetof(Vertex, m_color);

	attribute_descriptions[2].binding  = 0;
	attribute_descriptions[2].location = 2;
	attribute_descriptions[2].format   = VK_FORMAT_R32G32_SFLOAT;
	attribute_descriptions[2].offset   = offsetof(Vertex, m_tex_coord);

	return attribute_descriptions;
}

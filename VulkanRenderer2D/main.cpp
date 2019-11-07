#include <thread>
#include "renderer.h"

static void framebuffer_resize_callback(GLFWwindow* window, int width, int height)
{
	reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window))->signal_framebuffer_resize();
}

const std::vector<Renderer::Vertex> vertices = {
	{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
	{{0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
	{{0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
	{{-0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
	{{-0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
	{{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}}
};

int main()
{
	int width = 800;
	int height = 600;
	glm::vec2 camera_position(0.0f, 0.0f);

	try
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		GLFWwindow* window = glfwCreateWindow(width, height, "Vulkan Renderer", nullptr, nullptr);

		glfwSetFramebufferSizeCallback(window, framebuffer_resize_callback);

		Renderer renderer(window);

		glfwSetWindowUserPointer(window, &renderer);

		renderer.create_model(vertices, "../Vulkan/textures/Sneaking-white-dog.jpg");
		//renderer.create_model(vertices);

		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();

			int state = glfwGetKey(window, GLFW_KEY_D);
			if (state == GLFW_PRESS)
			{
				camera_position.x += 0.1f;
			}

			state = glfwGetKey(window, GLFW_KEY_A);
			if (state == GLFW_PRESS)
			{
				camera_position.x -= 0.1f;
			}

			state = glfwGetKey(window, GLFW_KEY_W);
			if (state == GLFW_PRESS)
			{
				camera_position.y += 0.1f;
			}

			state = glfwGetKey(window, GLFW_KEY_S);
			if (state == GLFW_PRESS)
			{
				camera_position.y -= 0.1f;
			}

			std::this_thread::sleep_for(std::chrono::milliseconds(20));

			renderer.draw_frame(camera_position);
		}

		renderer.wait_idle();

		glfwDestroyWindow(window);

		glfwTerminate();
	}
	catch (const std::exception& exception)
	{
		std::cerr << exception.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

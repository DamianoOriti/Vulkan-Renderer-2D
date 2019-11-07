#include "image.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Image::Image(const char* image_path) :
	m_data(nullptr)
{
	m_data = stbi_load(image_path, &m_width, &m_height, &m_channels, STBI_rgb_alpha);

	if (m_data == nullptr)
	{
		throw std::runtime_error("failed to load texture image!");
	}
}

Image::~Image()
{
	if (m_data != nullptr)
	{
		stbi_image_free(m_data);
		m_data = nullptr;
	}
}

const stbi_uc* Image::get_data()  const
{
	return m_data;
}

int Image::get_width()  const
{
	return m_width;
}

int Image::get_height()  const
{
	return m_height;
}

int Image::get_channels() const
{
	return m_channels;
}

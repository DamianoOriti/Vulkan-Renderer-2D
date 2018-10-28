#pragma once

#include <stdexcept>

typedef unsigned char stbi_uc;

class Image
{
public:
	Image(const char* filename);
	~Image();

	const stbi_uc* get_data() const;
	int get_width() const;
	int get_height() const;
	int get_channels() const;

private:
	stbi_uc* m_data;
	int m_width;
	int m_height;
	int m_channels;
};

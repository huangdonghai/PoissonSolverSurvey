
#include <algorithm>
#include <cmath>
#include "ImageHelper.h"

ByteImage::ByteImage(unsigned char* imageData, int width, int height, int channels)
{
	m_needFreeData = false;
	m_imageData = imageData;
	m_width = width;
	m_height = height;
	m_channels = channels;
}

ByteImage::ByteImage(int width, int height, int channels)
{
	m_needFreeData = true;
	m_width = width;
	m_height = height;
	m_channels = channels;
	m_imageData = new unsigned char[width * height * channels];
}

ByteImage::ByteImage(const ByteImage& rhs)
{
	m_width = rhs.m_width;
	m_height = rhs.m_height;
	m_channels = rhs.m_channels;
	int size = m_width * m_height * m_channels;
	m_needFreeData = false;
	m_imageData = nullptr;

	if (size > 0) {
		m_needFreeData = true;
		m_imageData = new unsigned char[size];
		memcpy(m_imageData, rhs.m_imageData, size);
	}
}

ByteImage::~ByteImage()
{
	if (m_needFreeData)
		delete[] m_imageData;
}

std::vector<float> ByteImage::Avarage()
{
	std::vector<float> ret(m_channels);

	float scale = 1.0f / (m_width * m_height);
	for (int y = 0; y < m_height; y++) {
		for (int x = 0; x < m_width; x++) {
			for (int channel = 0; channel < m_channels; channel++) {
				ret[channel] += Pixel(x, y, channel) * scale;
			}
		}
	}
	return ret;
}

bool ByteImage::PoissonFilter(ByteImage& dst, ByteImage& src)
{
	if (dst.m_width != src.m_width || dst.m_height != src.m_height || dst.m_channels != src.m_channels)
		return false;

	for (int y = 0; y < dst.m_height; y++) {
		for (int x = 0; x < dst.m_width; x++) {
			for (int channel = 0; channel < dst.m_channels; channel++) {
				int count = 0;

				int n = src.Pixel(x, y + 1, channel);
				int s = src.Pixel(x, y - 1, channel);
				int e = src.Pixel(x + 1, y, channel);
				int w = src.Pixel(x - 1, y, channel);
				int center = src.Pixel(x, y, channel);

				n = n >= 0 ? count++, n : 0;
				s = s >= 0 ? count++, s : 0;
				e = e >= 0 ? count++, e : 0;
				w = w >= 0 ? count++, w : 0;

				int value = count * center - (n + s + e + w);

				// map value to [0, 255]
				value = value < 0 ? - value : value;
				value = value < 0 ? 0 : value>255 ? 255 : value;

				dst.SetPixel(x, y, channel, value);
			}
		}
	}

	return true;
}

FloatImage::FloatImage(int width, int height, int channels, float initValues)
{
	m_needFreeData = true;
	m_width = width;
	m_height = height;
	m_channels = channels;
	m_imageData = new float[width * height * channels]{};

	if (initValues != 0) {
		for (int i = 0; i < m_width * m_height * m_channels; i++)
			m_imageData[i] = initValues;
	}
}

FloatImage::FloatImage(const FloatImage& rhs)
{
	m_width = rhs.m_width;
	m_height = rhs.m_height;
	m_channels = rhs.m_channels;
	int size = m_width * m_height * m_channels;
	m_needFreeData = false;
	m_imageData = nullptr;

	if (size > 0) {
		m_needFreeData = true;
		m_imageData = new float[size];
		memcpy(m_imageData, rhs.m_imageData, size * sizeof(float));
	}
}

FloatImage::~FloatImage()
{
	if (m_needFreeData)
		delete[] m_imageData;
}

unsigned char* FloatImage::ToBytes(float scale, std::vector<float>* offset, bool abs)
{
	if (!m_imageData)
		return nullptr;

	unsigned char* ret = new unsigned char[m_width * m_height * m_channels];
	for (int y = 0; y < m_height*m_width; y++) {
		for (int ch = 0; ch < m_channels; ch++) {
			auto f = m_imageData[y * m_channels + ch] * scale;
			if (offset) {
				f += (*offset)[ch];
			}
			int v = (int)(f);
			if (abs)
				v = v < 0 ? -v : v;
			v = v < 0 ? 0 : v > 255 ? 255 : v;
			ret[y * m_channels + ch] = v;
		}
	}
	return ret;
}

bool FloatImage::PoissonFilter(FloatImage& dst, ByteImage& src)
{
	if (dst.m_width != src.m_width || dst.m_height != src.m_height || dst.m_channels != src.m_channels)
		return false;

	for (int y = 0; y < dst.m_height; y++) {
		for (int x = 0; x < dst.m_width; x++) {
			for (int channel = 0; channel < dst.m_channels; channel++) {
				int count = 0;

				int n = src.Pixel(x, y + 1, channel);
				int s = src.Pixel(x, y - 1, channel);
				int e = src.Pixel(x + 1, y, channel);
				int w = src.Pixel(x - 1, y, channel);
				int center = src.Pixel(x, y, channel);

				n = n >= 0 ? count++, n : 0;
				s = s >= 0 ? count++, s : 0;
				e = e >= 0 ? count++, e : 0;
				w = w >= 0 ? count++, w : 0;

				int value = count * center - (n + s + e + w);

				// map value to [0, 255]
				//value = value < 0 ? -value : value;
				//value = value < 0 ? 0 : value>255 ? 255 : value;

				dst.SetPixel(x, y, channel, (float)value);
			}
		}
	}

	return true;
}

FloatImage FloatImage::JacobiCpu(int numIters)
{
	FloatImage temp1(m_width, m_height, m_channels);
	FloatImage temp2(m_width, m_height, m_channels);

	FloatImage *srcPtr = &temp1;
	FloatImage *dstPtr = &temp2;

	for (int it = 0; it < numIters; it++) {
		FloatImage &src = *srcPtr;
		FloatImage &dst = *dstPtr;

		for (int y = 0; y < dst.m_height; y++) {
			for (int x = 0; x < dst.m_width; x++) {
				for (int channel = 0; channel < dst.m_channels; channel++) {
					int count = 0;

					float n = src.Pixel(x, y + 1, channel);
					float s = src.Pixel(x, y - 1, channel);
					float e = src.Pixel(x + 1, y, channel);
					float w = src.Pixel(x - 1, y, channel);
					float center = Pixel(x, y, channel);

					n = !isnan(n) ? count++, n : 0;
					s = !isnan(s) ? count++, s : 0;
					e = !isnan(e) ? count++, e : 0;
					w = !isnan(w) ? count++, w : 0;

					float value = (center + (n + s + e + w)) / (float)count;

					// map value to [0, 255]
					//value = value < 0 ? -value : value;
					//value = value < 0 ? 0 : value>255 ? 255 : value;

					dst.SetPixel(x, y, channel, value);
				}
			}
		}
		std::swap(srcPtr, dstPtr);
	}

	return *srcPtr;
}

#include "cuda_runtime.h"
cudaError_t jacobiWithCuda(int numIters, int width, int height, int channels, const float* b, float* result, bool isSOR);
cudaError_t conjugateGradientWithCuda(int numIters, int width, int height, int channels, const float* hostb, float* hostresult);

FloatImage FloatImage::JacobiCuda(int numIters)
{
	FloatImage temp1(m_width, m_height, m_channels);
	jacobiWithCuda(numIters, m_width, m_height, m_channels, m_imageData, temp1.m_imageData, false);
	return temp1;
}
FloatImage FloatImage::SorCuda(int numIters)
{
	FloatImage temp1(m_width, m_height, m_channels);
	jacobiWithCuda(numIters, m_width, m_height, m_channels, m_imageData, temp1.m_imageData, true);
	return temp1;
}
FloatImage FloatImage::ConjugateGradientCuda(int numIters)
{
	FloatImage temp1(m_width, m_height, m_channels);
	conjugateGradientWithCuda(numIters, m_width, m_height, m_channels, m_imageData, temp1.m_imageData);
	return temp1;
}

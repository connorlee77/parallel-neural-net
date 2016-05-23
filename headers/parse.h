#ifndef PARSE_H
#define PARSE_H


float* read_mnist_labels(std::string full_path, int number_of_labels);
float** read_mnist_images(std::string full_path, int number_of_images, int image_size);

#endif // PARSE_H
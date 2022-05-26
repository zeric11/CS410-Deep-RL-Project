#include "c_DQN.h"


struct Input* create_input(int image_height, int image_width, int height_downscale_factor, int width_downscale_factor, int max_images_size) {
    struct Input* input = (struct Input*)malloc(sizeof(struct Input));

    input->image_height = image_height;
    input->image_width = image_width;
    input->height_downscale_factor = height_downscale_factor;
    input->width_downscale_factor = width_downscale_factor;
    input->max_images_size = max_images_size;
    input->image_size = (image_height / height_downscale_factor) * (image_width / width_downscale_factor);
    input->input_layer_size = input->image_size * max_images_size;

    input->images = (struct Image**)malloc(max_images_size * sizeof(struct Image*));
    input->current_images_size = 0;

    return input;
}


void add_image(struct Input* input, __uint8_t* rgb_values) {
    int rgb_height = input->image_height;
    int rgb_width = input->image_width;
    int height_scaling = input->height_downscale_factor;
    int width_scaling = input->width_downscale_factor;
    int image_height = rgb_height / height_scaling;
    int image_width = rgb_width / width_scaling;

    struct Image* image = (struct Image*)malloc(sizeof(struct Image));
    image->pixels = (double*)malloc(input->image_size * sizeof(double));
    image->size = input->image_size;

    for(register int i = 0; i < image_height; ++i) {
        for(register int j = 0; j < image_width; ++j) {
            double pixel_sum = 0;
            for(register int k = 0; k < height_scaling; ++k) {
                for(register int l = 0; l < width_scaling; ++l) {
                    int rgb_index = 3 * ((j * width_scaling) + (i * rgb_width * height_scaling) + (l) + (k * rgb_width));
                    pixel_sum += rgb_values[rgb_index] + rgb_values[rgb_index + 1] + rgb_values[rgb_index + 2];
                }
            }
            image->pixels[(i * image_width) + j] = pixel_sum / (height_scaling * width_scaling * 3 * 255);
        }
    }

    input->images[input->current_images_size] = image;
    ++input->current_images_size;

    //display_image(image, image_height, image_width);
}


double* create_input_layer(struct Input* input) {
    while(input->current_images_size < input->max_images_size) {
        input->images[input->current_images_size] = create_image_copy(input->images[input->current_images_size - 1]);
        ++input->current_images_size;
    }

    double* input_layer = (double*)malloc(input->input_layer_size * sizeof(double));
    for(register int i = 0; i < input->current_images_size; i += input->image_size) {
        struct Image* image = input->images[i];
        for(register int j = 0; j < image->size; ++j) {
            input_layer[i + j] = image->pixels[j];
        }
    }
    return input_layer;
}


void clear_images(struct Input* input) {
    if(input->images) {
        for(register int i = 0; i < input->current_images_size; ++i) {
            if(input->images[i]) {
                free_image(input->images[i]);
            }
            input->images[i] = NULL;
        }
    }
    input->current_images_size = 0;
}


void free_input(struct Input* input) {
    if(input) {
        if(input->images) {
            clear_images(input);
            free(input->images);
        }
        free(input);
    }
}


void free_image(struct Image* image) {
    if(image) {
        if(image->pixels) {
            free(image->pixels);
        }
        free(image);
    }
}


struct Image* create_image_copy(struct Image* src_image) {
    struct Image* new_image = (struct Image*)malloc(sizeof(struct Image));
    new_image->pixels = create_double_array_copy(src_image->pixels, src_image->size);
    new_image->size = src_image->size;
    return new_image;
}


void display_image(struct Image* image, int height, int width) {
    for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
            double value = image->pixels[(i * width) + j];
            if(value > 0) {
                printf("@");
            } else {
                printf(".");
            }
        }
        printf("\n");
    }
}
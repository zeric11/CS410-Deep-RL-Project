#include "c_DQN.h"


struct ConvLayer* create_input(int image_height, int image_width, int final_image_height, int final_image_width, int max_images_size) {
    struct ConvLayer* input = (struct ConvLayer*)malloc(sizeof(struct ConvLayer));

    input->initial_image_height = image_height;
    input->initial_image_width = image_width;
    input->final_image_height = final_image_height;
    input->final_image_width = final_image_width;
    input->max_images_size = max_images_size;
    input->final_image_size = final_image_height * final_image_width;
    input->input_layer_size = input->final_image_size * max_images_size;

    input->images = (struct Image**)malloc(max_images_size * sizeof(struct Image*));
    input->current_images_size = 0;

    input->filter_amount = 0;
    input->filter_height = 0;
    input->filter_width = 0;
    input->filters = NULL;
    input->combined_filter = NULL;

    return input;
}


void add_filter(struct ConvLayer* conv_layer, double* filter, int height, int width) {
    conv_layer->filter_height = height;
    conv_layer->filter_width = width;
    int size = height * width;

    double** new_filters = (double**)malloc((conv_layer->filter_amount + 1) * sizeof(double*));
    for(int i = 0; i < conv_layer->filter_amount + 1; ++i) {
        new_filters[i] = (double*)malloc(size * sizeof(double));
    }
    copy_2D_double_array(new_filters, conv_layer->filters, conv_layer->filter_amount, size);
    copy_double_array(new_filters[conv_layer->filter_amount], filter, size);
    free_2D_double_array(conv_layer->filters, conv_layer->filter_amount);
    conv_layer->filters = new_filters;
    ++conv_layer->filter_amount;

    update_combined_filter(conv_layer);
}


void update_combined_filter(struct ConvLayer* conv_layer) {
    if(!conv_layer->combined_filter) {
        conv_layer->combined_filter = (double*)malloc(conv_layer->filter_height * conv_layer->filter_width * sizeof(double));
    }
    for(int i = 0; i < conv_layer->filter_amount; ++i) {
        for(int j = 0; j < conv_layer->filter_height * conv_layer->filter_width; ++j) {
            if(i == 0) {
                conv_layer->combined_filter[j] = 0;
            }
            conv_layer->combined_filter[j] += conv_layer->filters[i][j];
        }
    }
}


void add_image(struct ConvLayer* conv_layer, double* rgb_values) {
    int rgb_height = conv_layer->initial_image_height;
    int rgb_width = conv_layer->initial_image_width;
    int image_height = conv_layer->final_image_height;
    int image_width = conv_layer->final_image_width;

    if(conv_layer->filter_amount == 0) {
        conv_layer->images[conv_layer->current_images_size] = create_resized_image(conv_layer, rgb_values);
    } else {
        conv_layer->images[conv_layer->current_images_size] = create_filtered_image(conv_layer, rgb_values);
    }
    ++conv_layer->current_images_size;

    //display_image(conv_layer->images[conv_layer->current_images_size - 1]->pixels, image_height, image_width);
}


struct Image* create_resized_image(struct ConvLayer* conv_layer, double* rgb_values) {
    int initial_height = conv_layer->initial_image_height;
    int initial_width = conv_layer->initial_image_width;
    int final_height = conv_layer->final_image_height;
    int final_width = conv_layer->final_image_width;

    struct Image* image = (struct Image*)malloc(sizeof(struct Image));
    image->size = final_height * final_width;
    image->pixels = (double*)malloc(image->size * sizeof(double));

    int x_ratio = (int)((initial_width << 16) / final_width) + 1;
    int y_ratio = (int)((initial_height << 16) / final_height) + 1;

    for(int i = 0; i < final_height; ++i) {
        for(int j = 0; j < final_width; ++j) {
            int k = (j * x_ratio) >> 16;
            int l = (i * y_ratio) >> 16;
            int rgb_index = ((l * initial_width) + k) * 3;
            double r = rgb_values[rgb_index];
            double g = rgb_values[rgb_index + 1];
            double b = rgb_values[rgb_index + 2];
            image->pixels[(i * final_width) + j] = (r + g + b) / (3 * 255);
        }
    }

    return image;
}


double* create_resized_filtered_image(double* pixels, int initial_height, int initial_width, int final_height, int final_width) {
    double* resized_pixels = (double*)malloc(final_height * final_width * sizeof(double));

    int x_ratio = (int)((initial_width << 16) / final_width) + 1;
    int y_ratio = (int)((initial_height << 16) / final_height) + 1;
    for(int i = 0; i < final_height; ++i) {
        for(int j = 0; j < final_width; ++j) {
            int k = (j * x_ratio) >> 16;
            int l = (i * y_ratio) >> 16;
            resized_pixels[(i * final_width) + j] = pixels[(l * initial_width) + k];
        }
    }

    return resized_pixels;
}


struct CreateFilteredImageThreadParams {
    int starting_index;
    int job_size;
    int initial_width;
    int final_width;
    int filter_height;
    int filter_width;
    double height_ratio;
    double width_ratio;
    double* rgb_values;
    double* filter;
    double* filtered_image;

};


void* create_filtered_image_thread(void* params_ptr) {
    struct CreateFilteredImageThreadParams* params = (struct CreateFilteredImageThreadParams*)params_ptr;
    int starting_index = params->starting_index;
    int job_size = params->job_size;
    int initial_width = params->initial_width;
    int final_width = params->final_width;
    int filter_height = params->filter_height;
    int filter_width = params->filter_width;
    double height_ratio = params->height_ratio;
    double width_ratio = params->width_ratio;
    double* rgb_values = params->rgb_values;
    double* filter = params->filter;
    double* filtered_image = params->filtered_image;

    for(register int i = starting_index; i < starting_index + job_size; ++i) {
        for(int j = 0; j < final_width; ++j) {
            //int rgb_row_index = ((int)width_ratio >> 16) * i;
            //int rgb_col_index = ((int)height_ratio >> 16) * j;
            int rgb_row_index = (int)((double)i * height_ratio); 
            int rgb_col_index = (int)((double)j * width_ratio);
            double sum = 0;
            for(int k = 0; k < filter_height; ++k) {
                for(int l = 0; l < filter_width; ++l) {
                    int rgb_index = 3 * (((rgb_row_index + k) * initial_width) + (rgb_col_index + l));
                    int filter_index = (k * filter_width) + l;
                    double r = rgb_values[rgb_index];
                    double g = rgb_values[rgb_index + 1];
                    double b = rgb_values[rgb_index + 2];
                    sum += (r + g + b) * filter[filter_index];
                }
            }
            filtered_image[(i * final_width) + j] = sum / (double)(3 * 255 * filter_height * filter_width);  
        }
    }
}


struct Image* create_filtered_image(struct ConvLayer* conv_layer, double* rgb_values) {
    int initial_height = conv_layer->initial_image_height;
    int initial_width = conv_layer->initial_image_width;
    int final_height = conv_layer->final_image_height;
    int final_width = conv_layer->final_image_width;

    //int height_ratio = (int)((initial_height << 16) / final_height);
    //int width_ratio = (int)((initial_width << 16) / final_width);
    double height_ratio = (double)initial_height / (double)final_height;
    double width_ratio = (double)initial_width / (double)final_width;
    
    double* filtered_image = (double*)malloc(final_height * final_width * sizeof(double));

    int thread_amount = final_height < MAX_THREADS ? 1 : MAX_THREADS;
    int job_size = (int)(final_height / thread_amount);
    int last_job_size = final_height - (job_size * (thread_amount - 1));

    struct CreateFilteredImageThreadParams params_list[thread_amount];
    pthread_t threads[thread_amount];

    for(register int i = 0; i < thread_amount; ++i) {
        params_list[i].starting_index = job_size * i;
        params_list[i].job_size = i == thread_amount - 1 ? last_job_size : job_size;
        params_list[i].initial_width = conv_layer->initial_image_width;
        params_list[i].final_width = final_width;
        params_list[i].filter_height = conv_layer->filter_height;
        params_list[i].filter_width = conv_layer->filter_width;
        params_list[i].height_ratio = height_ratio;
        params_list[i].width_ratio = width_ratio;
        params_list[i].rgb_values = rgb_values;
        params_list[i].filter = conv_layer->combined_filter;
        params_list[i].filtered_image = filtered_image;
        pthread_create(&threads[i], NULL, create_filtered_image_thread, &params_list[i]);
    }

    for(register int i = 0; i < thread_amount; ++i) {
        pthread_join(threads[i], NULL);
    }

    struct Image* image = (struct Image*)malloc(sizeof(struct Image));
    image->pixels = filtered_image;
    image->size = conv_layer->final_image_size;
    return image;
}

/*
struct Image* create_filtered_image(struct ConvLayer* conv_layer, double* rgb_values) {
    int filter_height = conv_layer->filter_height;
    int filter_width = conv_layer->filter_width;
    int initial_height = conv_layer->initial_image_height;
    int initial_width = conv_layer->initial_image_width;
    //int filtered_height = initial_height - filter_height + 1;
    //int filtered_width = initial_width - filter_width + 1;
    int final_height = conv_layer->final_image_height;
    int final_width = conv_layer->final_image_width;

    //int height_ratio = (int)((initial_height << 16) / final_height) + 1;
    //int width_ratio = (int)((initial_width << 16) / final_width) + 1;
    double height_ratio = (double)initial_height / (double)final_height;
    double width_ratio = (double)initial_width / (double)final_width;
    
    double* filtered_image = (double*)malloc(final_height * final_width * sizeof(double));

    for(int i = 0; i < final_height; ++i) {
        for(int j = 0; j < final_width; ++j) {
            int rgb_row_index = (int)((double)i * height_ratio); //(int)(width_ratio * i) >> 16;
            int rgb_col_index = (int)((double)j * width_ratio);//(int)(height_ratio * j) >> 16;
            double sum = 0;
            for(int k = 0; k < filter_height; ++k) {
                for(int l = 0; l < filter_width; ++l) {
                    int rgb_index = 3 * (((rgb_row_index + k) * initial_width) + (rgb_col_index + l));
                    int filter_index = (k * filter_width) + l;
                    double r = rgb_values[rgb_index];
                    double g = rgb_values[rgb_index + 1];
                    double b = rgb_values[rgb_index + 2];
                    sum += (r + g + b) * conv_layer->combined_filter[filter_index];
                }
            }
            filtered_image[(i * final_width) + j] = sum / (double)(3 * 255 * filter_height * filter_width);  
        }
    }

    struct Image* image = (struct Image*)malloc(sizeof(struct Image));
    image->pixels = filtered_image;
    image->size = conv_layer->final_image_size;
    return image;
}
*/
/*
struct Image* create_filtered_image(struct ConvLayer* conv_layer, double* rgb_values) {
    int filter_height = conv_layer->filter_height;
    int filter_width = conv_layer->filter_width;
    int initial_height = conv_layer->initial_image_height;
    int initial_width = conv_layer->initial_image_width;
    int filtered_height = initial_height - filter_height + 1;
    int filtered_width = initial_width - filter_width + 1;
    int final_height = conv_layer->final_image_height;
    int final_width = conv_layer->final_image_width;

    double* filtered_image = (double*)malloc(filtered_height * filtered_width * sizeof(double));
    for(int i = 0; i < filtered_height; ++i) {
        for(int j = 0; j < filtered_width; ++j) {
            double sum = 0;
            for(int k = 0; k < filter_height; ++k) {
                for(int l = 0; l < filter_width; ++l) {
                    int rgb_index = 3 * (((i + k) * initial_width) + (j + l));
                    int filter_index = (k * filter_width) + l;
                    double r = rgb_values[rgb_index];
                    double g = rgb_values[rgb_index + 1];
                    double b = rgb_values[rgb_index + 2];
                    sum += (r + g + b) * conv_layer->combined_filter[filter_index];
                }
            }
            filtered_image[(i * filtered_width) + j] = sum / (3 * 255 * filter_height * filter_width);  
        }
    }

    struct Image* image = (struct Image*)malloc(sizeof(struct Image));
    image->pixels = create_resized_filtered_image(filtered_image, filtered_height, filtered_width, final_height, final_width);
    image->size = conv_layer->final_image_size;
    free(filtered_image);
    return image;
}
*/


double* create_input_layer(struct ConvLayer* conv_layer) {
    while(conv_layer->current_images_size < conv_layer->max_images_size) {
        conv_layer->images[conv_layer->current_images_size] = create_image_copy(conv_layer->images[conv_layer->current_images_size - 1]);
        ++conv_layer->current_images_size;
    }

    double* input_layer = (double*)malloc(conv_layer->input_layer_size * sizeof(double));
    for(register int i = 0; i < conv_layer->current_images_size; ++i) {
        struct Image* image = conv_layer->images[i];
        for(register int j = 0; j < image->size; ++j) {
            input_layer[(i * conv_layer->final_image_size) + j] = image->pixels[j];
        }
    }

    //display_image(input_layer, conv_layer->final_image_height * conv_layer->current_images_size, conv_layer->final_image_width);

    return input_layer;
}


void clear_images(struct ConvLayer* conv_layer) {
    if(conv_layer->images) {
        for(register int i = 0; i < conv_layer->current_images_size; ++i) {
            if(conv_layer->images[i]) {
                free_image(conv_layer->images[i]);
            }
            conv_layer->images[i] = NULL;
        }
    }
    conv_layer->current_images_size = 0;
}


void free_input(struct ConvLayer* conv_layer) {
    if(conv_layer) {
        if(conv_layer->images) {
            clear_images(conv_layer);
            free(conv_layer->images);
        }
        if(conv_layer->filters) {
            free_2D_double_array(conv_layer->filters, conv_layer->filter_amount);
        }
        if(conv_layer->combined_filter) {
            free(conv_layer->combined_filter);
        }
        free(conv_layer);
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


void display_image(double* pixels, int height, int width) {
    for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
            double value = pixels[(i * width) + j];
            if(value > 0) {
                printf("@");
            } else if(value < 0) {
                printf("$");
            } else {
                printf(".");
            }
        }
        printf("\n");
    }
}


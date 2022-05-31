#include "c_DQN.h"


struct ConvLayer* create_conv_layer(int image_height, int image_width, int final_image_height, int final_image_width, int max_images_amount) {
    struct ConvLayer* conv_layer = (struct ConvLayer*)malloc(sizeof(struct ConvLayer));

    conv_layer->initial_image_height = image_height;
    conv_layer->initial_image_width = image_width;
    conv_layer->final_image_height = final_image_height;
    conv_layer->final_image_width = final_image_width;
    conv_layer->max_images_amount = max_images_amount;
    conv_layer->final_image_size = final_image_height * final_image_width;

    conv_layer->images = (struct Image**)malloc(max_images_amount * sizeof(struct Image*));
    conv_layer->current_images_amount = 0;

    conv_layer->filter_amount = 0;
    conv_layer->filter_height = 0;
    conv_layer->filter_width = 0;
    conv_layer->filters = NULL;

    return conv_layer;
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
    conv_layer->final_image_size = conv_layer->final_image_height * conv_layer->final_image_width * conv_layer->filter_amount;
}


void add_image(struct ConvLayer* conv_layer, double* rgb_values) {
    int rgb_height = conv_layer->initial_image_height;
    int rgb_width = conv_layer->initial_image_width;
    int filter_amount = conv_layer->filter_amount;
    int image_height = conv_layer->final_image_height * (filter_amount > 0 ? filter_amount : 1);
    int image_width = conv_layer->final_image_width;

    if(conv_layer->filter_amount == 0) {
        conv_layer->images[conv_layer->current_images_amount] = create_resized_image(conv_layer, rgb_values);
    } else {
        conv_layer->images[conv_layer->current_images_amount] = create_filtered_image(conv_layer, rgb_values);
    }
    ++conv_layer->current_images_amount;

    //display_image(conv_layer->images[conv_layer->current_images_size - 1]->pixels, image_height, image_width);
}


struct CreateResizedImageThreadParams {
    int starting_index;
    int job_size;
    int initial_width;
    int final_width;
    double height_ratio;
    double width_ratio;
    double* rgb_values;
    double* resized_image;
};


void* create_resized_image_thread(void* params_ptr) {
    struct CreateResizedImageThreadParams* params = (struct CreateResizedImageThreadParams*)params_ptr;
    int starting_index = params->starting_index;
    int job_size = params->job_size;
    int initial_width = params->initial_width;
    int final_width = params->final_width;
    int height_ratio = params->height_ratio;
    int width_ratio = params->width_ratio;
    double* rgb_values = params->rgb_values;
    double* resized_image = params->resized_image;

    for(register int i = starting_index; i < starting_index + job_size; ++i) {
        for(int j = 0; j < final_width; ++j) {
            int row_index = (i * height_ratio) >> 16;
            int col_index = (j * width_ratio) >> 16;
            int rgb_index = ((row_index * initial_width) + col_index) * 3;
            double r = rgb_values[rgb_index];
            double g = rgb_values[rgb_index + 1];
            double b = rgb_values[rgb_index + 2];
            resized_image[(i * final_width) + j] = (0.2990 * r) + (0.5870 * g) + (0.1140 * b) / 255;
        }
    }
}


struct Image* create_resized_image(struct ConvLayer* conv_layer, double* rgb_values) {
    int initial_height = conv_layer->initial_image_height;
    int initial_width = conv_layer->initial_image_width;
    int final_height = conv_layer->final_image_height;
    int final_width = conv_layer->final_image_width;

    struct Image* image = (struct Image*)malloc(sizeof(struct Image));
    image->size = final_height * final_width;
    image->pixels = (double*)malloc(image->size * sizeof(double));
    
    int height_ratio = (int)((initial_height << 16) / final_height) + 1;
    int width_ratio = (int)((initial_width << 16) / final_width) + 1;
    
    int thread_amount = final_height < MAX_THREADS ? 1 : MAX_THREADS;
    int job_size = (int)(final_height / thread_amount);
    int last_job_size = final_height - (job_size * (thread_amount - 1));
    struct CreateResizedImageThreadParams params_list[thread_amount];
    pthread_t threads[thread_amount];
    for(register int i = 0; i < thread_amount; ++i) {
        params_list[i].starting_index = job_size * i;
        params_list[i].job_size = i == thread_amount - 1 ? last_job_size : job_size;
        params_list[i].initial_width = conv_layer->initial_image_width;
        params_list[i].final_width = final_width;
        params_list[i].height_ratio = height_ratio;
        params_list[i].width_ratio = width_ratio;
        params_list[i].rgb_values = rgb_values;
        params_list[i].resized_image = image->pixels;
        pthread_create(&threads[i], NULL, create_resized_image_thread, &params_list[i]);
    }
    for(register int i = 0; i < thread_amount; ++i) {
        pthread_join(threads[i], NULL);
    }

    return image;
}


struct CreateFilteredImageThreadParams {
    int initial_height;
    int initial_width;
    int final_height;
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
    int initial_height = params->initial_height;
    int initial_width = params->initial_width;
    int final_height = params->final_height;
    int final_width = params->final_width;
    int filter_height = params->filter_height;
    int filter_width = params->filter_width;
    double height_ratio = params->height_ratio;
    double width_ratio = params->width_ratio;
    double* rgb_values = params->rgb_values;
    double* filter = params->filter;
    double* filtered_image = params->filtered_image;

    for(register int i = 0; i < final_height; ++i) {
        for(int j = 0; j < final_width; ++j) {
            //int rgb_row_index = ((int)width_ratio >> 16) * i;
            //int rgb_col_index = ((int)height_ratio >> 16) * j;
            int rgb_row_index = (int)((double)i * height_ratio); 
            int rgb_col_index = (int)((double)j * width_ratio);
            if(rgb_row_index > initial_height - filter_height - 1) {
                rgb_row_index = initial_height - filter_height - 1;
            }
            if(rgb_col_index > initial_width - filter_width - 1) {
                rgb_col_index = initial_width - filter_width - 1;
            }
            
            double sum = 0;
            for(int k = 0; k < filter_height; ++k) {
                for(int l = 0; l < filter_width; ++l) {
                    int rgb_index = 3 * (((rgb_row_index + k) * initial_width) + (rgb_col_index + l));
                    int filter_index = (k * filter_width) + l;
                    double r = rgb_values[rgb_index];
                    double g = rgb_values[rgb_index + 1];
                    double b = rgb_values[rgb_index + 2];
                    sum += ((0.2990 * r) + (0.5870 * g) + (0.1140 * b)) * filter[filter_index];
                }
            }
            filtered_image[(i * final_width) + j] = sum / (double)(255 * filter_height * filter_width);  
        }
    }
}


struct Image* create_filtered_image(struct ConvLayer* conv_layer, double* rgb_values) {
    int initial_height = conv_layer->initial_image_height;
    int initial_width = conv_layer->initial_image_width;
    int final_height = conv_layer->final_image_height;
    int final_width = conv_layer->final_image_width;
    int image_size = final_height * final_width;
    int final_image_size = conv_layer->final_image_size;
    int filter_amount = conv_layer->filter_amount;

    double* filtered_image = (double*)malloc(final_image_size * sizeof(double));

    //int height_ratio = (int)((initial_height << 16) / final_height);
    //int width_ratio = (int)((initial_width << 16) / final_width);
    double height_ratio = (double)initial_height / (double)final_height;
    double width_ratio = (double)initial_width / (double)final_width;

    struct CreateFilteredImageThreadParams params_list[filter_amount];
    pthread_t threads[filter_amount];
    for(register int i = 0; i < filter_amount; ++i) {
        params_list[i].initial_height = initial_height;
        params_list[i].initial_width = initial_width;
        params_list[i].final_height = final_height;
        params_list[i].final_width = final_width;
        params_list[i].filter_height = conv_layer->filter_height;
        params_list[i].filter_width = conv_layer->filter_width;
        params_list[i].height_ratio = height_ratio;
        params_list[i].width_ratio = width_ratio;
        params_list[i].rgb_values = rgb_values;
        params_list[i].filter = conv_layer->filters[i];
        params_list[i].filtered_image = filtered_image + (image_size * i);
        pthread_create(&threads[i], NULL, create_filtered_image_thread, &params_list[i]);
    }
    for(register int i = 0; i < filter_amount; ++i) {
        pthread_join(threads[i], NULL);
    }

    struct Image* image = (struct Image*)malloc(sizeof(struct Image));
    image->pixels = filtered_image;
    image->size = conv_layer->final_image_size;
    return image;
}


struct CreateInputLayerThreadParams {
    double* input_layer;
    struct Image* image;
};


void* create_input_layer_thread(void* params_ptr) {
    struct CreateInputLayerThreadParams* params = (struct CreateInputLayerThreadParams*)params_ptr;
    double* input_layer = params->input_layer;
    struct Image* image = params->image;

    for(register int j = 0; j < image->size; ++j) {
        input_layer[j] = image->pixels[j];
    }
}


double* create_input_layer(struct ConvLayer* conv_layer) {
    while(conv_layer->current_images_amount < conv_layer->max_images_amount) {
        conv_layer->images[conv_layer->current_images_amount] = create_image_copy(conv_layer->images[conv_layer->current_images_amount - 1]);
        ++conv_layer->current_images_amount;
    }

    int input_layer_size = conv_layer->final_image_size * conv_layer->current_images_amount;
    double* input_layer = (double*)malloc(input_layer_size * sizeof(double));

    struct CreateInputLayerThreadParams params_list[conv_layer->current_images_amount];
    pthread_t threads[conv_layer->current_images_amount];
    for(register int i = 0; i < conv_layer->current_images_amount; ++i) {
        params_list[i].input_layer = input_layer + (conv_layer->final_image_size * i);
        params_list[i].image = conv_layer->images[i];
        pthread_create(&threads[i], NULL, create_input_layer_thread, &params_list[i]);
    }
    for(register int i = 0; i < conv_layer->current_images_amount; ++i) {
        pthread_join(threads[i], NULL);
    }

    //display_image(input_layer, input_layer_size / conv_layer->final_image_width, conv_layer->final_image_width);

    return input_layer;
}


void clear_images(struct ConvLayer* conv_layer) {
    if(conv_layer->images) {
        for(register int i = 0; i < conv_layer->current_images_amount; ++i) {
            if(conv_layer->images[i]) {
                free_image(conv_layer->images[i]);
            }
            conv_layer->images[i] = NULL;
        }
    }
    conv_layer->current_images_amount = 0;
}


void free_conv_layer(struct ConvLayer* conv_layer) {
    if(conv_layer) {
        if(conv_layer->images) {
            clear_images(conv_layer);
            free(conv_layer->images);
        }
        if(conv_layer->filters) {
            free_2D_double_array(conv_layer->filters, conv_layer->filter_amount);
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


# Install and load required packages
install.packages("keras")
install.packages("tensorflow")
install.packages("reticulate")

library(keras)
library(tensorflow)
library(reticulate)

# Load the Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()

# Split into training and test sets
train_images <- fashion_mnist$train$x
train_labels <- fashion_mnist$train$y
test_images <- fashion_mnist$test$x
test_labels <- fashion_mnist$test$y

# Normalize the images to values between 0 and 1
train_images <- train_images / 255
test_images <- test_images / 255

# Reshape the images for the CNN
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))

# Build the CNN Model
model <- keras_model_sequential()

# Add layers to the model
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile Model
model %>% compile(
  optimizer = optimizer_adam(),
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Train Model
history <- model %>% fit(
  x = train_images,         # Explicitly naming the argument
  y = train_labels,         # Explicitly naming the argument
  epochs = 10,
  batch_size = 64,
  validation_split = 0.2
)

# Make predictions

# Select two images from the test set
sample_images <- test_images[1:2,,,drop=FALSE]  # Ensuring dimensions are preserved

# Predict the class
predictions <- model %>% predict(sample_images)

# Output predictions
predicted_classes <- apply(predictions, 1, which.max) - 1  # R uses 1-based indexing, so subtract 1
print(predicted_classes)

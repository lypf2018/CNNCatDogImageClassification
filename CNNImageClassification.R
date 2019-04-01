# download package "EBImage"
# source("http://bioconductor.org/biocLite.R")
# biocLite("EBImage")

# library(EBImage)
require(jpeg)
require(RCurl)
library(keras)

dog_path <- "http://www.utdallas.edu/~yxs173830/train/dogs/dog%20("
cat_path <- "http://www.utdallas.edu/~yxs173830/train/cats/cat%20("
data <- list()
label <- list()
train_dog_num <- 678
train_cat_num <- 761
for(i in 1:train_dog_num){
  data[[i]] <- readJPEG(getURLContent(paste(dog_path,i,").jpg", sep = ""), binary=TRUE))
  label[i] <- 0
}

for(i in 1:train_cat_num){
  data[[678+i]] <- readJPEG(getURLContent(paste(cat_path,i,").jpg", sep = ""), binary=TRUE))
  label[678+i] <- 1
}

label <- array(as.numeric(unlist(label)))

# Resize
for (i in 1:1439) {data[[i]] <- resize(data[[i]],50,50)}

# Reshape
for (i in 1:1439) {data[[i]] <- array(data[[i]], dim = c(50, 50, 3))}

display(data[[8]])

# split data
train_image <- list()
index <- sample(1:length(data),round(0.8*length(data)))
count <- 1
for (i in index) {train_image[[count]] <- data[[i]]
count <- count+1}
train_label <- as.array(label[index])

test_index <- setdiff(c(1:1439),index)
test_image <- list()
count <- 1
for (i in test_index) {test_image[[count]] <- data[[i]]
count <- count+1}
test_label <- as.array(label[-index])

train_image_array<-array(dim = c(length(index),50,50,3))
for (i in 1:length(index)) {
  for (j in 1:50) {
    for (k in 1:50) {
      for (l in 1:3) {
        train_image_array[i,j,k,l]<-train_image[[i]][j,k,l]
      }
    }
  }
}

test_image_array<-array(dim = c(length(test_index),50,50,3))
for (i in 1:length(test_index)) {
  for (j in 1:50) {
    for (k in 1:50) {
      for (l in 1:3) {
        test_image_array[i,j,k,l]<-test_image[[i]][j,k,l]
      }
    }
  }
}

train_label <- array(as.numeric(unlist(train_label)))
test_label <- array(as.numeric(unlist(test_label)))

#model

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(50, 50, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")

model <- model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history<-model %>% fit(
  train_image_array, train_label, 
  epochs = 10, batch_size=64
)

results <- model %>% evaluate(test_image_array, test_label)
results

# predict_classes(model,test_image_array)


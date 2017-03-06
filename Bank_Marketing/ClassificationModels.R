library(caret)

data <- read.csv("./Data/bank-additional-full.csv")

# Randomly order the dataset
rows <- sample(nrow(data))
data <- data[rows, ]

# Find row to split on
split <- round(nrow(data) * .70)
train <- data[1:split, ]
test <- data[(split + 1): nrow(data), ]

# Confirm test set size
nrow(train) / nrow(data)

# Fit glm model: model
model <- glm(y ~ ., family = "binomial"(link="logit"), train)

# Predict on test: p
p <- predict(model, test, type = "response")

# Calculate class probabilities: p_class
p_class <- ifelse(p > 0.90, "Yes", "No")

# Create confusion matrix
table(p_class, test[["y"]])
confusionMatrix(p_class, test[["y"]])

# Make ROC curve
colAUC(p, test[["y"]], plotROC = TRUE)

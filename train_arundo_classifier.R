# train_arundo_classifier.R
#
# Trains and tests Random Forest (ranger) classifiers and a regression of
# Arundo donax cover along the Santa Clara River from Landsat phenology, NAIP,
# and LiDAR predictors.
#
# Inputs:
#   input_maps/arundo_coverage.kmz  Ground-truth polygons, one KML layer per
#                                   Arundo cover class (attribute ARDO_class).
#   Landsat GV phenology raster     24 bands (~2 week windows) of green
#                                   vegetation fractional cover at 30 m.
#   derived_predictors/naip_30m.tif   12 NAIP bands (mean and spatial SD of 6
#   derived_predictors/lidar_30m.tif  spectral values) and 31 LiDAR bands
#                                     (point density + 30 height bins), both on
#                                     the 30 m grid. Created by
#                                     build_predictor_rasters.R.
#
# Outputs (written to output/):
#   feature_set_comparison.csv  Accuracy and Kappa for each predictor subset.
#   multiclass_<feature_set>_confusion_matrix.csv, ..._accuracy.csv,
#   ..._variable_importance.csv  (one set per feature subset)
#   multiclass_all_map.tif       Map from the all-predictor multiclass model.
#   binary_<class>_confusion_matrix.csv, binary_<class>_accuracy.csv,
#   binary_<class>_variable_importance.csv, binary_<class>_map.tif
#       (one per class threshold, all predictors; if kRunBinaryClassifiers)
#   regression_<stratified|unstratified>_metrics.csv
#       R^2, RMSE, MAE (percent cover), Spearman rank correlation with the true
#       class, and weighted Kappa of binned predictions, per split.
#   regression_<stratified|unstratified>_confusion_matrix.csv, ..._accuracy.csv
#       Test-set predictions binned back into the cover classes.
#   regression_<stratified|unstratified>_variable_importance.csv, ..._map.tif
#       (map in percent cover; if kRunCoverRegression)
#   All accuracy tables include unweighted, linear-weighted, and
#   quadratic-weighted Kappa (weights treat the classes as ordered).
#
# Method summary:
#   1. Load each cover-class layer from the KMZ.
#   2. Stack all predictors on the 30 m grid and extract every 30 m pixel whose
#      center falls inside a polygon (many samples per polygon). Pixels missing
#      any predictor (e.g. outside LiDAR coverage) are dropped for ALL feature
#      subsets so that the subsets are compared on identical samples.
#   3. Split POLYGONS (not pixels) 70/15/15 into train/validation/test,
#      stratified by class, so no polygon leaks across sets. Polygons left with
#      no samples are excluded before splitting.
#   4. Train a multiclass classifier on each predictor subset, then a set of
#      binary threshold classifiers on all predictors. By default
#      (kStratifyTrainingByClass) the TRAINING pixels are subsampled to the same
#      number per class (the size of the rarest class), so rare high-cover
#      classes are not swamped by "<1". Validation and test sets are NOT
#      resampled, so their accuracies reflect the natural class frequencies.
#      The very rare ">95" class is excluded from the analysis entirely.
#   5. Optionally treat cover as pseudo-continuous: each sampled pixel gets a
#      random percent cover drawn uniformly from its class range (e.g. 75-95 for
#      "75-95"), and a random forest regression predicts it from all predictors.
#      It is run twice, with and without stratification (which, if used,
#      balances the original classes).
#   6. Report confusion matrices, accuracies, and Kappa on the test set, and
#      predict maps from the all-predictor models.
#
# Assumptions to confirm (see notes in the accompanying message):
#   - Binary classifiers label a pixel "Low" when its class is at or below the
#     threshold class (e.g. "1-5" = classes <1 and 1-5) and "High" otherwise.
#     When stratifying, Low and High are balanced against each other.
#   - The validation set is used to report validation accuracy only; no
#     hyperparameter tuning is performed yet.

library(sf)
library(terra)
library(ranger)

# ---- Configuration ----------------------------------------------------------

kArundoKmzPath <- "input_maps/arundo_coverage.kmz"
kPhenologyPath <- "/mnt/e/birds/scr/imagery/backward_window_GV_phenology_Santa_Clara_2018.tif"
kPredictorDirectory <- "derived_predictors"
kOutputDirectory <- "output"
kSplitFractions <- c(train = 0.70, validation = 0.15, test = 0.15)
kRandomSeed <- 42
# If TRUE, training pixels are sampled evenly from each class (see header).
kStratifyTrainingByClass <- TRUE
# Classes dropped from the analysis (too few polygons to stratify against).
kExcludedClasses <- ">95"
# Which secondary models to run after the multiclass feature-set comparison.
kRunBinaryClassifiers <- TRUE
kRunCoverRegression <- TRUE

# Cover classes in increasing order; names match the KML folder names.
kClassOrder <- c("<1", "1-5", "5-10", "10-25", "25-50", "50-75", "75-95", ">95")
# Percent cover range (lower, upper) of each class, for the regression.
kClassCoverRangePercent <- list(
  "<1" = c(0, 1), "1-5" = c(1, 5), "5-10" = c(5, 10), "10-25" = c(10, 25),
  "25-50" = c(25, 50), "50-75" = c(50, 75), "75-95" = c(75, 95),
  ">95" = c(95, 100))

# ---- Load ground truth ------------------------------------------------------

LoadArundoPolygons <- function(kmz_path) {
  # Returns one sf data frame of all class polygons with a factor column
  # `arundo_class` and an integer `polygon_id`.
  extraction_directory <- tempfile("arundo_kmz_")
  dir.create(extraction_directory)
  unzip(kmz_path, exdir = extraction_directory)
  kml_path <- file.path(extraction_directory, "doc.kml")

  # Each cover class is its own KML layer; the "Feature Labels" layer holds
  # label points and is skipped.
  layer_names <- st_layers(kml_path)$name
  class_layer_names <- intersect(kClassOrder, layer_names)
  message("Found cover-class layers: ", paste(class_layer_names, collapse = ", "))

  polygons <- do.call(rbind, lapply(class_layer_names, function(layer_name) {
    layer <- st_read(kml_path, layer = layer_name, quiet = TRUE)
    layer <- st_zm(layer)  # Drop the constant z=0 coordinate.
    layer$arundo_class <- layer_name
    layer[, "arundo_class"]
  }))
  polygons$arundo_class <- factor(polygons$arundo_class, levels = kClassOrder)
  polygons$polygon_id <- seq_len(nrow(polygons))
  polygons
}

# ---- Sample pixels within polygons -----------------------------------------

ExtractPixelsWithinPolygons <- function(landsat_scale_raster, polygons) {
  # Extracts every 30 m pixel whose center lies inside a polygon. Returns a
  # data frame with polygon_id, arundo_class, and one column per band. Pixels
  # with any missing band are dropped.
  polygons_in_raster_crs <- st_transform(polygons, crs(landsat_scale_raster))
  samples <- extract(landsat_scale_raster, vect(polygons_in_raster_crs))
  # extract() returns the polygon's row index in column `ID`.
  samples$polygon_id <- polygons_in_raster_crs$polygon_id[samples$ID]
  samples$arundo_class <- polygons_in_raster_crs$arundo_class[samples$ID]
  samples$ID <- NULL
  na.omit(samples)
}

LoadPredictorStack <- function() {
  # Returns one SpatRaster on the NAIP-extent 30 m grid with the 24 phenology
  # bands (gv_period_01..24), the NAIP bands (naip_*) and the LiDAR bands
  # (lidar_*). LiDAR covers a smaller area, so it is padded with NA elsewhere.
  naip <- rast(file.path(kPredictorDirectory, "naip_30m.tif"))
  lidar <- crop(extend(rast(file.path(kPredictorDirectory, "lidar_30m.tif")), naip),
                naip)
  phenology <- crop(rast(kPhenologyPath), naip)
  names(phenology) <- sprintf("gv_period_%02d", seq_len(nlyr(phenology)))
  c(phenology, naip, lidar)
}

# ---- Split by polygon -------------------------------------------------------

AssignPolygonSplits <- function(polygons) {
  # Randomly assigns each polygon to train/validation/test, stratified by class
  # so every class appears in each set where it has enough polygons.
  set.seed(kRandomSeed)
  polygons$split <- NA_character_
  for (class_name in levels(polygons$arundo_class)) {
    class_rows <- which(polygons$arundo_class == class_name)
    shuffled_rows <- sample(class_rows)
    count <- length(shuffled_rows)
    train_count <- round(count * kSplitFractions[["train"]])
    validation_count <- round(count * kSplitFractions[["validation"]])
    split_labels <- c(rep("train", train_count),
                      rep("validation", validation_count))
    split_labels <- c(split_labels,
                      rep("test", count - length(split_labels)))
    polygons$split[shuffled_rows] <- split_labels
  }
  polygons
}

# ---- Accuracy assessment ----------------------------------------------------

ComputeAccuracy <- function(observed, predicted) {
  # observed / predicted: factors with identical levels.
  # Returns list(confusion_matrix, accuracy_table, overall_accuracy, kappa,
  # linear_weighted_kappa, quadratic_weighted_kappa). The weighted Kappas treat
  # levels as ordered, so near misses are penalized less than distant ones.
  confusion_matrix <- table(Predicted = predicted, Observed = observed)
  total <- sum(confusion_matrix)
  diagonal <- diag(confusion_matrix)
  row_totals <- rowSums(confusion_matrix)  # Predicted counts.
  column_totals <- colSums(confusion_matrix)  # Observed counts.

  users_accuracy <- diagonal / row_totals  # 1 - commission error.
  producers_accuracy <- diagonal / column_totals  # 1 - omission error.
  # Balanced accuracy per class: mean of sensitivity and specificity in a
  # one-vs-rest view of that class.
  true_negatives <- total - row_totals - column_totals + diagonal
  specificity <- true_negatives / (total - column_totals)
  balanced_accuracy <- (producers_accuracy + specificity) / 2

  overall_accuracy <- sum(diagonal) / total
  expected_agreement <- sum(row_totals * column_totals) / total^2
  kappa <- (overall_accuracy - expected_agreement) / (1 - expected_agreement)

  # Weighted Kappa = 1 - sum(w * observed) / sum(w * expected), where w is the
  # disagreement weight between predicted class i and observed class j.
  level_count <- nrow(confusion_matrix)
  class_distance <- abs(outer(seq_len(level_count), seq_len(level_count), "-")) /
    (level_count - 1)
  expected_counts <- outer(row_totals, column_totals) / total
  WeightedKappa <- function(weights) {
    1 - sum(weights * confusion_matrix) / sum(weights * expected_counts)
  }
  linear_weighted_kappa <- WeightedKappa(class_distance)
  quadratic_weighted_kappa <- WeightedKappa(class_distance^2)

  accuracy_table <- data.frame(
    class = rownames(confusion_matrix),
    users_accuracy = as.numeric(users_accuracy),
    producers_accuracy = as.numeric(producers_accuracy),
    balanced_accuracy = as.numeric(balanced_accuracy)
  )
  list(confusion_matrix = confusion_matrix, accuracy_table = accuracy_table,
       overall_accuracy = overall_accuracy, kappa = kappa,
       linear_weighted_kappa = linear_weighted_kappa,
       quadratic_weighted_kappa = quadratic_weighted_kappa)
}

EvaluateAndWrite <- function(model, samples, response_name, output_prefix) {
  # Reports validation and test accuracy; writes test-set tables to CSV.
  # Returns list(validation, test) of ComputeAccuracy() results.
  accuracy_by_split <- list()
  for (split_name in c("validation", "test")) {
    split_samples <- samples[samples$split == split_name, ]
    predicted <- predict(model, data = split_samples)$predictions
    # Align levels in case training dropped a class absent from its split.
    predicted <- factor(predicted, levels = levels(split_samples[[response_name]]))
    accuracy <- ComputeAccuracy(split_samples[[response_name]], predicted)
    accuracy_by_split[[split_name]] <- accuracy
    message(sprintf(paste("%s [%s]: overall accuracy %.3f, kappa %.3f,",
                          "weighted kappa %.3f (linear) %.3f (quadratic)"),
                    output_prefix, split_name, accuracy$overall_accuracy,
                    accuracy$kappa, accuracy$linear_weighted_kappa,
                    accuracy$quadratic_weighted_kappa))
    if (split_name == "test") WriteAccuracyTables(accuracy, output_prefix)
  }
  accuracy_by_split
}

WriteAccuracyTables <- function(accuracy, output_prefix) {
  # Writes a ComputeAccuracy() result as <prefix>_confusion_matrix.csv (rows =
  # predicted, columns = observed) and <prefix>_accuracy.csv.
  write.csv(as.data.frame.matrix(accuracy$confusion_matrix),
            file.path(kOutputDirectory,
                      paste0(output_prefix, "_confusion_matrix.csv")))
  accuracy_table <- rbind(
    accuracy$accuracy_table,
    data.frame(class = c("overall", "kappa", "linear_weighted_kappa",
                         "quadratic_weighted_kappa"),
               users_accuracy = c(accuracy$overall_accuracy, accuracy$kappa,
                                  accuracy$linear_weighted_kappa,
                                  accuracy$quadratic_weighted_kappa),
               producers_accuracy = NA, balanced_accuracy = NA))
  write.csv(accuracy_table,
            file.path(kOutputDirectory, paste0(output_prefix, "_accuracy.csv")),
            row.names = FALSE)
}

EvaluateRegressionAndWrite <- function(model, samples, response_name,
                                       output_prefix) {
  # Reports validation and test R^2, RMSE, and MAE against the pseudo-continuous
  # cover; the Spearman rank correlation between predicted cover and the true
  # class (the within-class random values carry no information, so ranks are
  # compared against the class); and the accuracy of predictions binned back
  # into cover classes. Writes <prefix>_metrics.csv and the test-set binned
  # class tables.
  # Returns list(validation, test) of binned-class ComputeAccuracy() results.
  class_names <- levels(samples$arundo_class)
  # A prediction falls in the class whose range contains it (lower bound
  # inclusive); values beyond the outer classes go to the outer classes.
  class_upper_bounds <- vapply(kClassCoverRangePercent[class_names], `[`,
                               numeric(1), 2)
  class_midpoints <- vapply(kClassCoverRangePercent[class_names], mean, numeric(1))
  RSquared <- function(observed, predicted) {
    1 - sum((observed - predicted)^2) / sum((observed - mean(observed))^2)
  }

  accuracy_by_split <- list()
  metrics <- list()
  for (split_name in c("validation", "test")) {
    split_samples <- samples[samples$split == split_name, ]
    observed <- split_samples[[response_name]]
    predicted <- predict(model, data = split_samples)$predictions
    errors <- predicted - observed
    # R^2 from predicting each pixel's true class midpoint: the best a model
    # could do given the random within-class noise in the target.
    midpoint_r_squared <- RSquared(
      observed, class_midpoints[as.character(split_samples$arundo_class)])
    metrics[[split_name]] <- data.frame(
      split = split_name, r_squared = RSquared(observed, predicted),
      rmse = sqrt(mean(errors^2)), mae = mean(abs(errors)),
      r_squared_if_class_known = midpoint_r_squared)

    predicted_class <- factor(
      class_names[findInterval(predicted, head(class_upper_bounds, -1)) + 1],
      levels = class_names)
    accuracy <- ComputeAccuracy(split_samples$arundo_class, predicted_class)
    accuracy_by_split[[split_name]] <- accuracy
    metrics[[split_name]]$spearman_rank_correlation <- cor(
      predicted, as.integer(split_samples$arundo_class), method = "spearman")
    metrics[[split_name]]$binned_accuracy <- accuracy$overall_accuracy
    metrics[[split_name]]$binned_kappa <- accuracy$kappa
    metrics[[split_name]]$binned_linear_weighted_kappa <-
      accuracy$linear_weighted_kappa
    metrics[[split_name]]$binned_quadratic_weighted_kappa <-
      accuracy$quadratic_weighted_kappa
    message(sprintf(paste("%s [%s]: R^2 %.3f (%.3f if class known), RMSE %.1f,",
                          "Spearman %.3f, binned class accuracy %.3f,",
                          "kappa %.3f, weighted kappa %.3f (linear)",
                          "%.3f (quadratic)"),
                    output_prefix, split_name, metrics[[split_name]]$r_squared,
                    midpoint_r_squared, metrics[[split_name]]$rmse,
                    metrics[[split_name]]$spearman_rank_correlation,
                    accuracy$overall_accuracy, accuracy$kappa,
                    accuracy$linear_weighted_kappa,
                    accuracy$quadratic_weighted_kappa))
    if (split_name == "test") WriteAccuracyTables(accuracy, output_prefix)
  }
  write.csv(do.call(rbind, metrics),
            file.path(kOutputDirectory, paste0(output_prefix, "_metrics.csv")),
            row.names = FALSE)
  accuracy_by_split
}

# ---- Training ---------------------------------------------------------------

StratifiedSample <- function(training_samples, response_name) {
  # Returns training_samples with the same number of rows (randomly drawn
  # without replacement) from each level of the response: the size of the
  # smallest level.
  rows_by_level <- split(seq_len(nrow(training_samples)),
                         training_samples[[response_name]])
  samples_per_level <- min(lengths(rows_by_level))
  set.seed(kRandomSeed)
  selected_rows <- unlist(lapply(rows_by_level, function(rows) {
    rows[sample.int(length(rows), samples_per_level)]
  }))
  message(sprintf("Stratified training: %d samples per class (%d total)",
                  samples_per_level, length(selected_rows)))
  training_samples[selected_rows, ]
}

TrainAndMap <- function(samples, predictor_raster, response_name, output_prefix,
                        predict_map = FALSE,
                        stratify_training = kStratifyTrainingByClass,
                        stratify_by = response_name) {
  # Trains on the training split using the layers of predictor_raster as
  # predictors, evaluates, and optionally predicts a map. A numeric response
  # gives a regression forest, a factor a classification forest. stratify_by
  # names the column whose levels are balanced when stratifying (it is not
  # used as a predictor). Returns list(model, out_of_bag_accuracy, accuracy)
  # where accuracy is the list from EvaluateAndWrite() or
  # EvaluateRegressionAndWrite(); out_of_bag_accuracy is NA for regression.
  predictor_names <- names(predictor_raster)
  is_regression <- is.numeric(samples[[response_name]])
  training_samples <- samples[samples$split == "train", ]
  if (stratify_training) {
    training_samples <- StratifiedSample(training_samples, stratify_by)
  }
  training_samples <- training_samples[, c(response_name, predictor_names)]
  model <- ranger(
    formula = as.formula(paste(response_name, "~ .")),
    data = training_samples,
    num.trees = 500,
    importance = "impurity",
    seed = kRandomSeed
  )
  out_of_bag_accuracy <- if (is_regression) NA else 1 - model$prediction.error
  if (is_regression) {
    message(sprintf("%s [out-of-bag]: R^2 %.3f", output_prefix, model$r.squared))
  } else {
    message(sprintf("%s [out-of-bag]: accuracy %.3f", output_prefix,
                    out_of_bag_accuracy))
  }
  write.csv(data.frame(predictor = names(model$variable.importance),
                       importance = as.numeric(model$variable.importance)),
            file.path(kOutputDirectory,
                      paste0(output_prefix, "_variable_importance.csv")),
            row.names = FALSE)
  if (is_regression) {
    accuracy <- EvaluateRegressionAndWrite(model, samples, response_name,
                                           output_prefix)
  } else {
    accuracy <- EvaluateAndWrite(model, samples, response_name, output_prefix)
  }

  if (predict_map) {
    predict(predictor_raster, model, na.rm = TRUE,
            fun = function(model, data, ...) {
              predictions <- predict(model, data = data)$predictions
              # Class maps store the class index; regression maps store cover.
              if (is.factor(predictions)) as.integer(predictions) else predictions
            },
            filename = file.path(kOutputDirectory,
                                 paste0(output_prefix, "_map.tif")),
            overwrite = TRUE)
  }
  list(model = model, out_of_bag_accuracy = out_of_bag_accuracy,
       accuracy = accuracy)
}

# ---- Main -------------------------------------------------------------------

Main <- function() {
  dir.create(kOutputDirectory, showWarnings = FALSE)

  polygons <- LoadArundoPolygons(kArundoKmzPath)
  polygons <- polygons[!polygons$arundo_class %in% kExcludedClasses, ]
  polygons$arundo_class <- droplevels(polygons$arundo_class)
  analysis_classes <- levels(polygons$arundo_class)
  predictor_stack <- LoadPredictorStack()

  # Sample all predictors at once; pixels missing any predictor are dropped.
  samples <- ExtractPixelsWithinPolygons(predictor_stack, polygons)
  message("Polygons with at least one complete sample: ",
          length(unique(samples$polygon_id)), " of ", nrow(polygons))
  message("Polygons per class before / after dropping incomplete ones:")
  print(rbind(before = table(polygons$arundo_class),
              after = table(polygons$arundo_class[
                polygons$polygon_id %in% samples$polygon_id])))
  message("Sampled pixels per class:")
  print(table(samples$arundo_class))

  # Split polygons that actually have samples, then attach splits to pixels.
  polygons <- AssignPolygonSplits(
    polygons[polygons$polygon_id %in% samples$polygon_id, ])
  samples <- merge(samples,
                   st_drop_geometry(polygons)[, c("polygon_id", "split")],
                   by = "polygon_id")
  samples$polygon_id <- NULL  # Not a predictor.

  # Version 1: predict the Arundo cover class directly, for each predictor
  # subset, to see what each data source contributes.
  all_predictor_names <- names(predictor_stack)
  PredictorsStartingWith <- function(...) {
    all_predictor_names[grepl(paste0("^(", paste(c(...), collapse = "|"), ")"),
                              all_predictor_names)]
  }
  feature_sets <- list(
    phenology = PredictorsStartingWith("gv_"),
    naip = PredictorsStartingWith("naip_"),
    lidar = PredictorsStartingWith("lidar_"),
    phenology_naip = PredictorsStartingWith("gv_", "naip_"),
    naip_lidar = PredictorsStartingWith("naip_", "lidar_"),
    all = all_predictor_names)
  comparison <- do.call(rbind, lapply(names(feature_sets), function(set_name) {
    result <- TrainAndMap(samples, predictor_stack[[feature_sets[[set_name]]]],
                          "arundo_class", paste0("multiclass_", set_name),
                          predict_map = (set_name == "all"))
    data.frame(feature_set = set_name,
               predictor_count = length(feature_sets[[set_name]]),
               out_of_bag_accuracy = result$out_of_bag_accuracy,
               validation_accuracy = result$accuracy$validation$overall_accuracy,
               validation_kappa = result$accuracy$validation$kappa,
               test_accuracy = result$accuracy$test$overall_accuracy,
               test_kappa = result$accuracy$test$kappa,
               test_linear_weighted_kappa =
                 result$accuracy$test$linear_weighted_kappa,
               test_quadratic_weighted_kappa =
                 result$accuracy$test$quadratic_weighted_kappa)
  }))
  print(comparison)
  write.csv(comparison, file.path(kOutputDirectory, "feature_set_comparison.csv"),
            row.names = FALSE)

  # Version 2: one binary classifier per class threshold, all predictors. The
  # highest remaining class is not a threshold (everything would be "Low").
  if (kRunBinaryClassifiers) {
    for (class_name in analysis_classes[-length(analysis_classes)]) {
      threshold_index <- match(class_name, analysis_classes)
      samples$cover_relative_to_threshold <- factor(
        ifelse(as.integer(samples$arundo_class) <= threshold_index, "Low", "High"),
        levels = c("Low", "High"))
      binary_samples <- samples[, setdiff(names(samples), "arundo_class")]
      TrainAndMap(binary_samples, predictor_stack, "cover_relative_to_threshold",
                  paste0("binary_", gsub("[<>]", "", class_name)),
                  predict_map = TRUE)
    }
  }

  # Version 3: pseudo-continuous regression. Each pixel gets a random percent
  # cover drawn uniformly from its class range. It is run with and without
  # stratification by the original classes, which are kept only for that and
  # for scoring. Stratified training shifts the target mean toward high cover,
  # and regression forests shrink predictions toward that mean.
  if (kRunCoverRegression) {
    set.seed(kRandomSeed)
    cover_ranges <- do.call(rbind, kClassCoverRangePercent[
      as.character(samples$arundo_class)])
    samples$pseudo_cover_percent <- runif(nrow(samples), cover_ranges[, 1],
                                          cover_ranges[, 2])
    for (stratify_regression in c(TRUE, FALSE)) {
      TrainAndMap(samples, predictor_stack, "pseudo_cover_percent",
                  paste0("regression_", ifelse(stratify_regression, "stratified",
                                               "unstratified")),
                  predict_map = TRUE, stratify_training = stratify_regression,
                  stratify_by = "arundo_class")
    }
  }
}

if (sys.nframe() == 0) {
  Main()
}

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
# Outputs (written to output/). Accuracy tables are for kReportedSplit
# (validation while tuning; switch to test for the final assessment):
#   feature_set_comparison.csv  Accuracy and Kappa for each predictor subset.
#   multiclass_<feature_set>_confusion_matrix.csv, ..._accuracy.csv,
#   ..._variable_importance.csv  (one set per feature subset)
#   multiclass_all_map.tif       Map from the all-predictor multiclass model.
#   binary_<class>_confusion_matrix.csv, binary_<class>_accuracy.csv,
#   binary_<class>_variable_importance.csv, binary_<class>_map.tif
#       (one per class threshold, all predictors; if kRunBinaryClassifiers)
#   regression_<stratified|unstratified>_metrics.csv
#       R^2, RMSE, MAE (percent cover), Spearman rank correlation with the true
#       class, and weighted Kappa of binned predictions.
#   regression_<stratified|unstratified>_confusion_matrix.csv, ..._accuracy.csv
#       Predictions binned back into the cover classes.
#   regression_<stratified|unstratified>_variable_importance.csv, ..._map.tif
#       (map in percent cover; if kRunCoverRegression)
#   <model>_out_of_fold_map.tif, <model>_out_of_fold_confusion_matrix.csv,
#   <model>_out_of_fold_accuracy.csv (+ _metrics.csv for regressions)
#       For every mapped model (if kWriteOutOfFoldMaps): a map in which each
#       spatial block is predicted by a model that never saw that block, and
#       the accuracy of those predictions at the non-test sample pixels.
#   All accuracy tables include unweighted, linear-weighted, and
#   quadratic-weighted Kappa (weights treat the classes as ordered).
#
# Method summary:
#   1. Load each cover-class layer from the KMZ.
#   2. Stack all predictors on the 30 m grid and extract every 30 m pixel whose
#      center falls inside a polygon (many samples per polygon). Pixels missing
#      any predictor (e.g. outside LiDAR coverage) are dropped for ALL feature
#      subsets so that the subsets are compared on identical samples.
#   3. Split POLYGONS (not pixels) into train/validation/test so no polygon
#      leaks across sets. Polygons left with no samples are excluded first.
#      With kSplitMethod = "spatial_blocks" (default) the study area is cut
#      into square blocks (kSpatialBlockSizeMeters); each polygon belongs to
#      the block holding the mean position of its pixels, and whole blocks are
#      assigned to a split (~70/15/15 of the pixels), choosing among many
#      random assignments the one that gives every class its fair share of
#      polygons in every split. Neighboring polygons
#      therefore share a split, so spatial autocorrelation between adjacent
#      polygons does not inflate held-out accuracy. "random_polygons" instead
#      splits polygons 70/15/15 at random, stratified by class.
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
#   6. Report confusion matrices, accuracies, and Kappa on kReportedSplit, and
#      predict maps from the all-predictor models.
#   7. Out-of-fold maps: spatial blocks are dealt into kOutOfFoldCount folds.
#      For each fold, a model with the same settings is trained on the
#      non-test polygons that have no pixel in that fold, and predicts the map
#      cells (and sample pixels) in that fold. The resulting map's agreement
#      with the polygons is an honest view of skill, unlike the main map,
#      which reproduces the training polygons it was fit to. Test polygons are
#      never used, so they stay held out.
#
# Assumptions to confirm (see notes in the accompanying message):
#   - Binary classifiers label a pixel "Low" when its class is at or below the
#     threshold class (e.g. "1-5" = classes <1 and 1-5) and "High" otherwise.
#     When stratifying, Low and High are balanced against each other.
#   - No hyperparameter tuning is automated yet; results on kReportedSplit
#     guide manual choices, so the test set should only be read at the end.

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
# "spatial_blocks" or "random_polygons" (see header, step 3).
kSplitMethod <- "spatial_blocks"
# Side length of the square spatial blocks used for splits and folds (meters).
kSpatialBlockSizeMeters <- 3000
# Random block assignments tried; the one with the most even class balance
# across splits is kept (see AssignSpatialBlockSplits).
kSplitCandidateCount <- 2000
# Split whose accuracy is reported and written. Keep "validation" while
# tuning; switch to "test" only for the final assessment.
kReportedSplit <- "validation"
# If TRUE, every mapped model also gets an out-of-fold map (header, step 7).
kWriteOutOfFoldMaps <- TRUE
kOutOfFoldCount <- 5
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
  # data frame with polygon_id, arundo_class, one column per band, and the
  # pixel center x and y (raster CRS). Pixels with any missing band are
  # dropped.
  polygons_in_raster_crs <- st_transform(polygons, crs(landsat_scale_raster))
  samples <- extract(landsat_scale_raster, vect(polygons_in_raster_crs),
                     xy = TRUE)
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

BlockColumnAndRow <- function(x, y) {
  # Returns a two-column matrix with the integer column and row of the square
  # spatial block holding each (x, y) point (meters).
  cbind(floor(x / kSpatialBlockSizeMeters), floor(y / kSpatialBlockSizeMeters))
}

AssignSpatialBlockSplits <- function(polygon_table) {
  # polygon_table: data frame with block_id, arundo_class, and pixel_count per
  # polygon. Assigns whole blocks to train/validation/test so that about
  # kSplitFractions of the sampled PIXELS fall in each; every polygon takes
  # its block's split. Blocks are few and classes cluster in space, so a
  # single random assignment can leave a split with almost no polygons of a
  # rare (high-cover) class. kSplitCandidateCount random block orders are
  # therefore tried, and the one kept maximizes the worst ratio, over every
  # class and split, of polygons received to polygons expected (split
  # fraction x class polygon count). Returns polygon_table with `split`.
  pixels_per_block <- tapply(polygon_table$pixel_count, polygon_table$block_id,
                             sum)
  expected_polygons <- outer(table(polygon_table$arundo_class), kSplitFractions)
  SplitBlocks <- function(block_order) {
    # Each block goes to the split whose cumulative pixel share holds the
    # block's midpoint.
    ordered_pixels <- pixels_per_block[block_order]
    block_midpoint_share <- (cumsum(ordered_pixels) - ordered_pixels / 2) /
      sum(ordered_pixels)
    block_split <- names(kSplitFractions)[
      findInterval(block_midpoint_share, cumsum(kSplitFractions)) + 1]
    names(block_split) <- names(ordered_pixels)
    factor(unname(block_split[polygon_table$block_id]),
           levels = names(kSplitFractions))
  }
  set.seed(kRandomSeed)
  best_split <- NULL
  best_score <- -Inf
  for (candidate in seq_len(kSplitCandidateCount)) {
    candidate_split <- SplitBlocks(sample(length(pixels_per_block)))
    received_polygons <- table(polygon_table$arundo_class, candidate_split)
    score <- min(received_polygons / expected_polygons)
    if (score > best_score) {
      best_score <- score
      best_split <- candidate_split
    }
  }
  message(sprintf(paste("Best of %d block assignments: every class/split",
                        "holds at least %.0f%% of its expected polygons"),
                  kSplitCandidateCount, 100 * best_score))
  polygon_table$split <- as.character(best_split)
  polygon_table
}

MakeFoldLookup <- function(template_raster, sampled_x, sampled_y) {
  # Deals every spatial block over template_raster's extent into one of
  # kOutOfFoldCount folds. Blocks holding samples are dealt out evenly in
  # random order; the remaining blocks get random folds. Returns a function
  # of (x, y) giving the fold of each point.
  first_block <- as.vector(BlockColumnAndRow(xmin(template_raster),
                                             ymin(template_raster)))
  last_block <- as.vector(BlockColumnAndRow(xmax(template_raster),
                                            ymax(template_raster)))
  block_counts <- last_block - first_block + 1
  BlockIndex <- function(x, y) sweep(BlockColumnAndRow(x, y), 2, first_block) + 1
  set.seed(kRandomSeed)
  fold_by_block <- matrix(
    sample.int(kOutOfFoldCount, prod(block_counts), replace = TRUE),
    block_counts[1], block_counts[2])
  sampled_blocks <- unique(BlockIndex(sampled_x, sampled_y))
  fold_by_block[sampled_blocks] <- sample(
    rep_len(seq_len(kOutOfFoldCount), nrow(sampled_blocks)))
  function(x, y) fold_by_block[BlockIndex(x, y)]
}

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

ReportClassAccuracy <- function(label, accuracy) {
  message(sprintf(paste("%s: overall accuracy %.3f, kappa %.3f,",
                        "weighted kappa %.3f (linear) %.3f (quadratic)"),
                  label, accuracy$overall_accuracy, accuracy$kappa,
                  accuracy$linear_weighted_kappa,
                  accuracy$quadratic_weighted_kappa))
}

EvaluateAndWrite <- function(model, samples, response_name, output_prefix) {
  # Reports and writes the accuracy on the kReportedSplit pixels. Returns the
  # ComputeAccuracy() result.
  split_samples <- samples[samples$split == kReportedSplit, ]
  predicted <- predict(model, data = split_samples)$predictions
  # Align levels in case training dropped a class absent from its split.
  predicted <- factor(predicted, levels = levels(split_samples[[response_name]]))
  accuracy <- ComputeAccuracy(split_samples[[response_name]], predicted)
  ReportClassAccuracy(sprintf("%s [%s]", output_prefix, kReportedSplit),
                      accuracy)
  WriteAccuracyTables(accuracy, output_prefix)
  accuracy
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

ScoreRegression <- function(scored_samples, response_name, predicted,
                            split_label) {
  # Scores predicted cover against the pseudo-continuous cover: R^2, RMSE, MAE;
  # the Spearman rank correlation between predicted cover and the true class
  # (the within-class random values carry no information, so ranks are
  # compared against the class); and the accuracy of predictions binned back
  # into cover classes. split_label names the scored pixels in the output.
  # Returns list(metrics = one-row data frame, accuracy = binned
  # ComputeAccuracy() result).
  class_names <- levels(scored_samples$arundo_class)
  # A prediction falls in the class whose range contains it (lower bound
  # inclusive); values beyond the outer classes go to the outer classes.
  class_upper_bounds <- vapply(kClassCoverRangePercent[class_names], `[`,
                               numeric(1), 2)
  class_midpoints <- vapply(kClassCoverRangePercent[class_names], mean,
                            numeric(1))
  RSquared <- function(observed, predicted) {
    1 - sum((observed - predicted)^2) / sum((observed - mean(observed))^2)
  }
  observed <- scored_samples[[response_name]]
  errors <- predicted - observed
  predicted_class <- factor(
    class_names[findInterval(predicted, head(class_upper_bounds, -1)) + 1],
    levels = class_names)
  accuracy <- ComputeAccuracy(scored_samples$arundo_class, predicted_class)
  metrics <- data.frame(
    split = split_label, r_squared = RSquared(observed, predicted),
    rmse = sqrt(mean(errors^2)), mae = mean(abs(errors)),
    # R^2 from predicting each pixel's true class midpoint: the best a model
    # could do given the random within-class noise in the target.
    r_squared_if_class_known = RSquared(
      observed, class_midpoints[as.character(scored_samples$arundo_class)]),
    spearman_rank_correlation = cor(
      predicted, as.integer(scored_samples$arundo_class), method = "spearman"),
    binned_accuracy = accuracy$overall_accuracy,
    binned_kappa = accuracy$kappa,
    binned_linear_weighted_kappa = accuracy$linear_weighted_kappa,
    binned_quadratic_weighted_kappa = accuracy$quadratic_weighted_kappa)
  list(metrics = metrics, accuracy = accuracy)
}

ReportAndWriteRegression <- function(scores, label, output_prefix) {
  # Prints ScoreRegression() results and writes <prefix>_metrics.csv and the
  # binned class tables.
  with(scores$metrics, message(sprintf(
    paste("%s: R^2 %.3f (%.3f if class known), RMSE %.1f, Spearman %.3f,",
          "binned class accuracy %.3f, kappa %.3f, weighted kappa %.3f",
          "(linear) %.3f (quadratic)"),
    label, r_squared, r_squared_if_class_known, rmse,
    spearman_rank_correlation, binned_accuracy, binned_kappa,
    binned_linear_weighted_kappa, binned_quadratic_weighted_kappa)))
  WriteAccuracyTables(scores$accuracy, output_prefix)
  write.csv(scores$metrics,
            file.path(kOutputDirectory, paste0(output_prefix, "_metrics.csv")),
            row.names = FALSE)
}

EvaluateRegressionAndWrite <- function(model, samples, response_name,
                                       output_prefix) {
  # Scores the regression on the kReportedSplit pixels (see ScoreRegression)
  # and writes the results. Returns the binned-class ComputeAccuracy() result.
  split_samples <- samples[samples$split == kReportedSplit, ]
  predicted <- predict(model, data = split_samples)$predictions
  scores <- ScoreRegression(split_samples, response_name, predicted,
                            kReportedSplit)
  ReportAndWriteRegression(scores, sprintf("%s [%s]", output_prefix,
                                           kReportedSplit), output_prefix)
  scores$accuracy
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

FitForest <- function(training_samples, response_name, predictor_names,
                      stratify_training, stratify_by) {
  # Fits a ranger forest of response_name on predictor_names, after balancing
  # the levels of stratify_by if stratify_training. A numeric response gives a
  # regression forest, a factor a classification forest.
  if (stratify_training) {
    training_samples <- StratifiedSample(training_samples, stratify_by)
  }
  ranger(
    formula = as.formula(paste(response_name, "~ .")),
    data = training_samples[, c(response_name, predictor_names)],
    num.trees = 500,
    importance = "impurity",
    seed = kRandomSeed
  )
}

PredictionsAsValues <- function(predictions, class_levels) {
  # Class predictions become their index in class_levels (as stored in class
  # maps); regression predictions are returned unchanged.
  if (is.factor(predictions)) {
    match(as.character(predictions), class_levels)
  } else {
    predictions
  }
}

WriteOutOfFold <- function(samples, predictor_raster, response_name,
                           output_prefix, stratify_training, stratify_by,
                           fold_raster) {
  # Blocked cross-validation over the non-test polygons (header, step 7).
  # samples must hold polygon_id and map_fold (the fold of each pixel's
  # block); fold_raster holds the fold of every map cell, in layer map_fold.
  # Writes <prefix>_out_of_fold_map.tif and the out-of-fold accuracy tables.
  predictor_names <- names(predictor_raster)
  class_levels <- levels(samples[[response_name]])  # NULL for regression.
  cv_samples <- samples[samples$split != "test", ]
  out_of_fold_predictions <- rep(NA_real_, nrow(cv_samples))
  models <- lapply(seq_len(kOutOfFoldCount), function(fold) {
    # Drop every polygon with any pixel in this fold, so no polygon that
    # reaches into the fold's blocks trains the model that predicts them.
    polygons_in_fold <- unique(cv_samples$polygon_id[cv_samples$map_fold == fold])
    training_samples <- cv_samples[!cv_samples$polygon_id %in% polygons_in_fold, ]
    FitForest(training_samples, response_name, predictor_names,
              stratify_training, stratify_by)
  })
  for (fold in seq_len(kOutOfFoldCount)) {
    fold_rows <- which(cv_samples$map_fold == fold)
    out_of_fold_predictions[fold_rows] <- PredictionsAsValues(
      predict(models[[fold]], data = cv_samples[fold_rows, ])$predictions,
      class_levels)
  }

  out_of_fold_prefix <- paste0(output_prefix, "_out_of_fold")
  label <- sprintf("%s [out-of-fold, %d folds, non-test pixels]",
                   output_prefix, kOutOfFoldCount)
  if (is.null(class_levels)) {
    ReportAndWriteRegression(
      ScoreRegression(cv_samples, response_name, out_of_fold_predictions,
                      "out_of_fold"),
      label, out_of_fold_prefix)
  } else {
    accuracy <- ComputeAccuracy(
      cv_samples[[response_name]],
      factor(class_levels[out_of_fold_predictions], levels = class_levels))
    ReportClassAccuracy(label, accuracy)
    WriteAccuracyTables(accuracy, out_of_fold_prefix)
  }

  predict(c(predictor_raster, fold_raster), models, na.rm = TRUE,
          fun = function(models, data, ...) {
            values <- rep(NA_real_, nrow(data))
            for (fold in seq_along(models)) {
              fold_rows <- which(data$map_fold == fold)
              if (length(fold_rows) == 0) next
              values[fold_rows] <- PredictionsAsValues(
                predict(models[[fold]], data = data[fold_rows, , drop = FALSE])$predictions,
                class_levels)
            }
            values
          },
          filename = file.path(kOutputDirectory,
                               paste0(out_of_fold_prefix, "_map.tif")),
          overwrite = TRUE)
  invisible(NULL)
}

TrainAndMap <- function(samples, predictor_raster, response_name, output_prefix,
                        predict_map = FALSE,
                        stratify_training = kStratifyTrainingByClass,
                        stratify_by = response_name, fold_raster = NULL) {
  # Trains on the training split using the layers of predictor_raster as
  # predictors, evaluates on kReportedSplit, and optionally predicts a map. A
  # numeric response gives a regression forest, a factor a classification
  # forest. stratify_by names the column whose levels are balanced when
  # stratifying (it is not used as a predictor). If predict_map and
  # fold_raster is given, an out-of-fold map is written too (WriteOutOfFold).
  # Returns list(model, out_of_bag_accuracy, accuracy) where accuracy is the
  # result of EvaluateAndWrite() or EvaluateRegressionAndWrite();
  # out_of_bag_accuracy is NA for regression.
  predictor_names <- names(predictor_raster)
  is_regression <- is.numeric(samples[[response_name]])
  model <- FitForest(samples[samples$split == "train", ], response_name,
                     predictor_names, stratify_training, stratify_by)
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
    class_levels <- levels(samples[[response_name]])
    predict(predictor_raster, model, na.rm = TRUE,
            fun = function(model, data, ...) {
              # Class maps store the class index; regression maps store cover.
              PredictionsAsValues(predict(model, data = data)$predictions,
                                  class_levels)
            },
            filename = file.path(kOutputDirectory,
                                 paste0(output_prefix, "_map.tif")),
            overwrite = TRUE)
    if (!is.null(fold_raster)) {
      WriteOutOfFold(samples, predictor_raster, response_name, output_prefix,
                     stratify_training, stratify_by, fold_raster)
    }
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
  # Each polygon is placed at the mean position of its sampled pixels.
  polygon_table <- aggregate(cbind(x, y) ~ polygon_id, data = samples,
                             FUN = mean)
  polygon_table$pixel_count <- as.vector(
    table(samples$polygon_id)[as.character(polygon_table$polygon_id)])
  polygon_table$arundo_class <- polygons$arundo_class[
    match(polygon_table$polygon_id, polygons$polygon_id)]
  polygon_block <- BlockColumnAndRow(polygon_table$x, polygon_table$y)
  polygon_table$block_id <- paste(polygon_block[, 1], polygon_block[, 2])
  polygon_table <- switch(kSplitMethod,
                          spatial_blocks = AssignSpatialBlockSplits(polygon_table),
                          random_polygons = AssignPolygonSplits(polygon_table),
                          stop("Unknown kSplitMethod: ", kSplitMethod))
  samples <- merge(samples, polygon_table[, c("polygon_id", "split")],
                   by = "polygon_id")
  message(sprintf("Split method: %s (%d blocks of %d m with samples)",
                  kSplitMethod, length(unique(polygon_table$block_id)),
                  kSpatialBlockSizeMeters))
  message("Polygons per class and split:")
  print(table(polygon_table$arundo_class, polygon_table$split))
  message("Sampled pixels per class and split:")
  print(table(samples$arundo_class, samples$split))
  # polygon_id, x, y (and map_fold below) stay in samples for the out-of-fold
  # maps; TrainAndMap() only passes the predictor columns to the forest.

  fold_raster <- NULL
  if (kWriteOutOfFoldMaps) {
    FoldAt <- MakeFoldLookup(predictor_stack, samples$x, samples$y)
    samples$map_fold <- FoldAt(samples$x, samples$y)
    fold_raster <- rast(predictor_stack, nlyrs = 1)
    cell_centers <- xyFromCell(fold_raster, seq_len(ncell(fold_raster)))
    values(fold_raster) <- FoldAt(cell_centers[, 1], cell_centers[, 2])
    names(fold_raster) <- "map_fold"
    message("Non-test sampled pixels per out-of-fold fold:")
    print(table(samples$map_fold[samples$split != "test"]))
  }

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
                          predict_map = (set_name == "all"),
                          fold_raster = fold_raster)
    data.frame(feature_set = set_name,
               predictor_count = length(feature_sets[[set_name]]),
               out_of_bag_accuracy = result$out_of_bag_accuracy,
               evaluated_split = kReportedSplit,
               accuracy = result$accuracy$overall_accuracy,
               kappa = result$accuracy$kappa,
               linear_weighted_kappa = result$accuracy$linear_weighted_kappa,
               quadratic_weighted_kappa =
                 result$accuracy$quadratic_weighted_kappa)
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
                  predict_map = TRUE, fold_raster = fold_raster)
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
                  stratify_by = "arundo_class", fold_raster = fold_raster)
    }
  }
}

if (sys.nframe() == 0) {
  Main()
}

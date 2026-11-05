# build_predictor_rasters.R
#
# Builds 30 m (Landsat-grid) predictor rasters from NAIP and LiDAR and caches
# them as GeoTIFFs so the classifier script does not have to re-read ~16 GB of
# imagery on every run.
#
# Inputs:
#   /mnt/e/birds/scr/imagery/backward_window_GV_phenology_Santa_Clara_2018.tif
#       Only used as the reference 30 m grid.
#   /mnt/e/birds/scr/NAIP/*.tif
#       RGB+NIR tiles, 0.6 m. Landsat cell edges fall exactly on the 0.6 m
#       lattice, so each 30 m cell is exactly 50 x 50 NAIP pixels.
#   /mnt/e/birds/scr/LiDAR/2018/vegetation_histograms/
#       vegetation_histogram_mosaic.tif  (10 m; bands 1-2 are discarded, bands
#                                         3-32 are the fraction of vegetation
#                                         points in each 1 m height bin, 0-30 m)
#       density_mosaic.tif               (10 m; vegetation point density)
#
# Outputs (written to derived_predictors/):
#   naip_30m.tif   12 bands: for each of 6 per-pixel quantities (red, green,
#                  blue, NIR divided by the mean DN across the 4 bands; NDVI;
#                  brightness = mean DN across the 4 bands), the mean and the
#                  spatial standard deviation across the 0.6 m pixels in the cell.
#   lidar_30m.tif  31 bands: lidar_point_density plus 30 height-bin fractions
#                  (lidar_height_bin_01 = 0-1 m, ... bin_30 = 29-30 m).
#
# Usage: Rscript build_predictor_rasters.R [naip|lidar|all] [max_blocks]
#   max_blocks limits the NAIP build to the first N blocks (for timing tests;
#   the result is then written to naip_30m_test.tif instead).
#
# NAD83 and WGS84 UTM 11N differ by roughly 1 m here, which is negligible at
# 30 m, so NAIP/LiDAR outputs are simply labeled with the phenology raster CRS.

library(sf)
library(terra)

kPhenologyPath <- "/mnt/e/birds/scr/imagery/backward_window_GV_phenology_Santa_Clara_2018.tif"
kNaipDirectory <- "/mnt/e/birds/scr/NAIP"
kLidarDirectory <- "/mnt/e/birds/scr/LiDAR/2018/vegetation_histograms"
kOutputDirectory <- "derived_predictors"
kNaipPixelsPerLandsatCell <- 50  # 30 m / 0.6 m
# NAIP is processed in blocks of Landsat cells to bound memory (4 x 400 cells =
# 200 x 20,000 NAIP pixels per block).
kBlockLandsatRows <- 4
kBlockLandsatColumns <- 400
kParallelWorkers <- 4  # More workers exhaust 10 GB of RAM.
kReadAttempts <- 3  # Reads from the mounted drive occasionally fail transiently.
kLidarHeightBinCount <- 30
kLidarDiscardedBandCount <- 2

# ---- NAIP -------------------------------------------------------------------

ComputeNaipBlock <- function(naip_paths, block_extent) {
  # Returns a 12-layer 30 m SpatRaster (6 means then 6 standard deviations) for
  # one block of Landsat cells, or NULL if the block has no NAIP data.
  naip_mosaic <- vrt(naip_paths)
  strip <- crop(naip_mosaic, block_extent)
  digital_numbers <- values(strip)  # One row per pixel, columns red..NIR.

  # Pixels outside every tile read as 0 in all bands; treat them as missing.
  is_empty <- rowSums(digital_numbers) == 0
  if (all(is_empty)) return(NULL)
  brightness <- rowMeans(digital_numbers)
  brightness[is_empty] <- NA
  red <- digital_numbers[, 1]
  nir <- digital_numbers[, 4]
  ndvi <- (nir - red) / (nir + red)  # NaN where nir + red == 0; handled below.
  ndvi[is_empty] <- NA

  per_pixel <- cbind(
    naip_red_normalized = red / brightness,
    naip_green_normalized = digital_numbers[, 2] / brightness,
    naip_blue_normalized = digital_numbers[, 3] / brightness,
    naip_nir_normalized = nir / brightness,
    naip_ndvi = ndvi,
    naip_brightness = brightness)
  per_pixel[is.nan(per_pixel)] <- NA

  per_pixel_raster <- rast(strip, nlyrs = ncol(per_pixel))
  values(per_pixel_raster) <- per_pixel
  names(per_pixel_raster) <- colnames(per_pixel)

  cell_mean <- aggregate(per_pixel_raster, fact = kNaipPixelsPerLandsatCell,
                         fun = "mean", na.rm = TRUE)
  cell_standard_deviation <- aggregate(per_pixel_raster,
                                       fact = kNaipPixelsPerLandsatCell,
                                       fun = "sd", na.rm = TRUE)
  names(cell_mean) <- paste0(colnames(per_pixel), "_mean")
  names(cell_standard_deviation) <- paste0(colnames(per_pixel), "_sd")
  c(cell_mean, cell_standard_deviation)
}

BuildNaipFeatures <- function(max_blocks = Inf) {
  naip_paths <- list.files(kNaipDirectory, pattern = "\\.tif$", full.names = TRUE)
  phenology_grid <- rast(kPhenologyPath)[[1]]
  # Landsat cells lying entirely within the NAIP mosaic.
  template <- crop(phenology_grid, ext(vrt(naip_paths)), snap = "in")
  message("NAIP template: ", nrow(template), " rows x ", ncol(template), " cols")

  # Enumerate blocks and skip those that touch no tile footprint.
  tile_extents <- lapply(naip_paths, function(path) ext(rast(path)))
  blocks <- list()
  for (first_row in seq(1, nrow(template), by = kBlockLandsatRows)) {
    last_row <- min(first_row + kBlockLandsatRows - 1, nrow(template))
    for (first_column in seq(1, ncol(template), by = kBlockLandsatColumns)) {
      last_column <- min(first_column + kBlockLandsatColumns - 1, ncol(template))
      block_extent <- ext(xFromCol(template, first_column) - 15,
                          xFromCol(template, last_column) + 15,
                          yFromRow(template, last_row) - 15,
                          yFromRow(template, first_row) + 15)
      if (any(vapply(tile_extents, relate, logical(1), y = block_extent,
                     relation = "intersects"))) {
        blocks[[length(blocks) + 1]] <- block_extent
      }
    }
  }
  blocks <- head(blocks, max_blocks)
  message(length(blocks), " blocks intersect NAIP tiles")

  # SpatRaster objects cannot be returned from forked workers, so each worker
  # writes its block to a temporary GeoTIFF and returns the file path.
  block_directory <- file.path(kOutputDirectory, "naip_blocks")
  dir.create(block_directory, showWarnings = FALSE)
  block_paths <- parallel::mclapply(seq_along(blocks), function(block_index) {
    for (attempt in seq_len(kReadAttempts)) {
      block <- try(ComputeNaipBlock(naip_paths, blocks[[block_index]]))
      if (!inherits(block, "try-error")) break
    }
    if (inherits(block, "try-error")) return(block)
    if (is.null(block)) return(NULL)
    block_path <- file.path(block_directory, sprintf("block_%04d.tif", block_index))
    writeRaster(block, block_path, overwrite = TRUE)
    block_path
  }, mc.cores = kParallelWorkers)
  failed <- vapply(block_paths, inherits, logical(1), what = "try-error")
  if (any(failed)) stop("NAIP blocks failed: ", sum(failed))
  block_paths <- unlist(Filter(Negate(is.null), block_paths))
  message("Mosaicking ", length(block_paths), " blocks")
  merged <- extend(vrt(block_paths), template)
  names(merged) <- names(rast(block_paths[1]))  # The VRT drops layer names.
  crs(merged) <- crs(template)

  output_path <- file.path(kOutputDirectory,
                           ifelse(is.finite(max_blocks), "naip_30m_test.tif",
                                  "naip_30m.tif"))
  writeRaster(merged, output_path, overwrite = TRUE)
  unlink(block_directory, recursive = TRUE)
  message("Wrote ", output_path)
}

# ---- LiDAR ------------------------------------------------------------------

BuildLidarFeatures <- function() {
  histogram <- rast(file.path(kLidarDirectory, "vegetation_histogram_mosaic.tif"))
  density <- rast(file.path(kLidarDirectory, "density_mosaic.tif"))
  histogram <- histogram[[(kLidarDiscardedBandCount + 1):nlyr(histogram)]]
  stopifnot(nlyr(histogram) == kLidarHeightBinCount)

  phenology_grid <- rast(kPhenologyPath)[[1]]
  # Project the extent of the LiDAR mosaic (NAD83) to the phenology CRS.
  lidar_extent <- ext(project(as.polygons(ext(density), crs = crs(density)),
                              crs(phenology_grid)))
  template <- crop(phenology_grid, lidar_extent, snap = "in")

  # A cell with zero vegetation points has an NaN histogram; call it zero in
  # every bin. NaN density means no LiDAR tile at all, and stays NaN.
  histogram <- ifel(is.na(histogram) & !is.na(density), 0, histogram)
  # Area-weight each 10 m cell by its point density so sparse cells do not
  # count as much as dense ones: weighted fraction = sum(f * d) / sum(d).
  weighted_histogram <- histogram * density
  mean_weighted <- project(weighted_histogram, template, method = "average")
  mean_density <- project(density, template, method = "average")
  fractions <- ifel(mean_density > 0, mean_weighted / mean_density, 0)

  names(fractions) <- sprintf("lidar_height_bin_%02d", seq_len(kLidarHeightBinCount))
  names(mean_density) <- "lidar_point_density"
  fractions <- mask(fractions, mean_density)  # Keep NaN where there is no LiDAR.
  features <- c(mean_density, fractions)
  crs(features) <- crs(template)

  output_path <- file.path(kOutputDirectory, "lidar_30m.tif")
  writeRaster(features, output_path, overwrite = TRUE)
  message("Wrote ", output_path)
}

# ---- Main -------------------------------------------------------------------

Main <- function() {
  arguments <- commandArgs(trailingOnly = TRUE)
  target <- ifelse(length(arguments) >= 1, arguments[1], "all")
  max_blocks <- ifelse(length(arguments) >= 2, as.numeric(arguments[2]), Inf)
  dir.create(kOutputDirectory, showWarnings = FALSE)
  if (target %in% c("naip", "all")) BuildNaipFeatures(max_blocks)
  if (target %in% c("lidar", "all")) BuildLidarFeatures()
}

if (sys.nframe() == 0) {
  Main()
}

# check_naip_coverage.R
#
# Checks whether every Arundo ground-truth polygon is covered by the NAIP tiles.
#
# Inputs:
#   input_maps/arundo_coverage.kmz  Ground-truth polygons (via LoadArundoPolygons).
#   /mnt/e/birds/scr/NAIP/*.tif    Tiled NAIP imagery (only headers are read).
#
# Output:
#   missing_polygons.gpkg  Polygons NOT fully inside the union of tile extents.
#                          Only written if there are any.
#
# Coverage is judged from tile extents (bounding boxes), not from valid pixels,
# so a polygon over a nodata collar inside a tile would count as covered.

library(sf)
library(terra)

source("train_arundo_classifier.R")  # For LoadArundoPolygons(); Main() is not run.

kNaipDirectory <- "/mnt/e/birds/scr/NAIP"
kMissingPolygonsPath <- "missing_polygons.gpkg"

CheckNaipCoverage <- function() {
  polygons <- LoadArundoPolygons(kArundoKmzPath)
  tile_paths <- list.files(kNaipDirectory, pattern = "\\.tif$", full.names = TRUE)

  # Build one footprint polygon per tile, all in the first tile's CRS.
  tile_footprints <- lapply(tile_paths, function(tile_path) {
    tile <- rast(tile_path)
    st_as_sfc(st_bbox(c(xmin = xmin(tile), xmax = xmax(tile),
                        ymin = ymin(tile), ymax = ymax(tile)),
                      crs = st_crs(crs(tile))))
  })
  tile_crs_list <- unique(vapply(tile_paths, function(path) crs(rast(path)),
                                 character(1)))
  message(length(tile_paths), " tiles; distinct CRS count: ",
          length(tile_crs_list))
  # Work in the (single) tile CRS with planar geometry; the spherical (s2)
  # engine rejects a few of the ground-truth polygons as invalid.
  sf_use_s2(FALSE)
  tile_crs <- st_crs(crs(rast(tile_paths[1])))
  naip_footprint <- st_union(do.call(c, tile_footprints))
  invalid_count <- sum(!st_is_valid(polygons))
  message("Invalid polygons repaired with st_make_valid: ", invalid_count)
  polygons <- st_make_valid(st_transform(polygons, tile_crs))

  fully_covered <- st_within(polygons, naip_footprint, sparse = FALSE)[, 1]
  touches_any <- st_intersects(polygons, naip_footprint, sparse = FALSE)[, 1]
  message("Polygons total: ", nrow(polygons),
          "; fully covered: ", sum(fully_covered),
          "; partially covered: ", sum(touches_any & !fully_covered),
          "; not covered at all: ", sum(!touches_any))

  missing_polygons <- polygons[!fully_covered, ]
  missing_polygons$coverage <- ifelse(touches_any[!fully_covered],
                                      "partial", "none")
  if (nrow(missing_polygons) > 0) {
    print(table(missing_polygons$arundo_class, missing_polygons$coverage))
    st_write(missing_polygons, kMissingPolygonsPath, delete_dsn = TRUE,
             quiet = TRUE)
    message("Wrote ", kMissingPolygonsPath)
  }
}

CheckNaipCoverage()

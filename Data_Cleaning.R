library(data.table)
library(tidyverse)
library(sf)
library(raster)

dat <- fread("SoilTrainingData.csv")
dat <- dat[!is.na(Latitude),]
dat[,Longitude := -1*Longitude]
datSf <- st_as_sf(dat, coords = c("Longitude","Latitude"), crs = 4326)
datSf <- datSf[datSf$NutrientRegime %in% c("A","B","C","D","E"),]
st_write(datSf,dsn = "TrainingData.gpkg", driver = "GPKG")
plot(datSf["NutrientRegime"],pch = 16)
table(dat$NutrientRegime)

testArea <- st_read(dsn = ".",layer = "TestArea3")
datSf <- st_transform(datSf,3005)
datSub <- datSf[testArea,]
library(sen2r)


## create dem layers
library(Rsagacmd)
dem <- raster("bc25fill")
aoi <- st_read(dsn = ".",layer = "TestArea3")
aoi <- st_transform(aoi,st_crs(dem))
dem2 <- crop(dem,aoi)

# Define inputs used in some of the SAGA tools below
reference_res <- res(dem_input)[1]
covariate_out <- out_path

## Scaling parameter for a few of the tools:
scale_param <- ifelse(
  round((reference_res*(25 / reference_res)) / reference_res) == 0, 1, 
  round((reference_res*(25 / reference_res)) / reference_res)
)

## Tool specific parameters
mrvbf_param <- 116.57 * (reference_res ^ -0.62)
tpi_param <- reference_res * 5
openness_param <- as.numeric(ncol(dem_input) / 2)
vrm_param <- 10
vall_ridg_param <- 7
mpi_param <- 50

slope = saga$ta_morphometry$slope_aspect_curvature(elevation = dem2,
                                                   c_gene = "gencurve",c_tota = "totcurve",method = 6,
                                                   unit_slope = 0,unit_aspect = 0)
writeRaster(slope$slope,"./DemLayers/Slope.tif", format = "GTiff")
writeRaster(slope$aspect,"./DemLayers/Aspect.tif", format = "GTiff")
flowAcc <- saga$ta_hydrology$flow_accumulation_recursive(elevation = dem2,flow_unit = 1,method = 4)
writeRaster(flowAcc$flow,"./DemLayers/FlowAccum.tif", format = "GTiff")
writeRaster(flowAcc$flow_length,"./DemLayers/FlowLength.tif", format = "GTiff")
tca <- saga$ta_hydrology$flow_accumulation_recursive(elevation = dem2,flow_unit = 1,method = 4)
twi <- saga$ta_hydrology$topographic_wetness_index_twi(slope = slope$slope,area = tca$flow,conv = 1,method = 1)
writeRaster(tca$flow,"./DemLayers/tca.tif", format = "GTiff")
writeRaster(twi,"./DemLayers/twi.tif", format = "GTiff")
mrvbf <- saga$ta_morphometry$multiresolution_index_of_valley_bottom_flatness_mrvbf(dem = dem2,t_slope = mrvbf_param,
                                                                                   t_pctl_v = 0.4, t_pctl_r = 0.35,
                                                                                   p_slope = 4,p_pctl = 3,update = F,
                                                                                   classify = F)
writeRaster(mrvbf$mrvbf,"./DemLayers/mrvbf.tif", format = "GTiff")
writeRaster(mrvbf$mrrtf,"./DemLayers/mrrtf.tif", format = "GTiff")
tri <- saga$ta_morphometry$terrain_ruggedness_index_tri(dem = dem2,mode = 0,
                                                        radius = scale_param,dw_weighting = 0)
writeRaster(tri,"./DemLayers/tri.tif", format = "GTiff")
tpi <- saga$ta_morphometry$topographic_position_index_tpi(dem = dem2,standard = T,
                                                          dw_weighting = 0)
writeRaster(tpi,"./DemLayers/tpi.tif",format = "GTiff")
val_depth <- saga$ta_channels$valley_depth(elevation = dem2,threshold = 1,
                                           nounderground = 1,order = 4)
writeRaster(val_depth$valley_depth,"./DemLayers/vdepth.tif", format = "GTiff")
##swi takes a long time
swi <- saga$ta_hydrology$saga_wetness_index(dem = dem2,suction = 10,area_type = 1,
                                            slope_type = 1,slope_min = 0,slope_off = 0,
                                            slope_weight = 1)

###soil layers
st_layers("./SoilMaps/BC_Soil_Map.gdb")
smap <- st_read("./SoilMaps/BC_Soil_Map.gdb",layer = "BC_Soil_Surveys")
smap <- st_cast(smap,"MULTIPOLYGON")
boundary <- st_as_sfc(st_bbox(testArea))
smap_small <- smap[boundary,]
smap_small <- st_buffer(smap_small, dist = 0)
smap_small <- st_intersection(smap_small,boundary)
st_write(smap_small,dsn = "ClippedSoil.gpkg")
layerDat <- st_drop_geometry(smap_small)

library(fasterize)
snames <- c("SOILCODE_1","DEVELOP1_1","PM1_1","Drain_1","TEXTURE_1","AWHC120_1")
soilStack <- stack()
for(lName in snames){
  temp <- smap_small[lName]
  colnames(temp)[1] <- "Var"
  temp$Var <- as.numeric(as.factor(temp$Var))
  temp <- st_cast(temp,"MULTIPOLYGON")
  rast <- fasterize(temp,raster = tpi,field = "Var")
  soilStack <- stack(soilStack,rast)
}

ndvi <- raster("./SatelliteLayers/data_mining_ndvi2-0000000000-0000032768.tif")
ndvi <- projectRaster(ndvi,tpi)
##load layers, extract points
layers <- c("Aspect.tif","FlowAccum.tif","FlowLength.tif","mrrtf.tif","mrvbf.tif",
            "Slope.tif","tca.tif","tpi.tif","tri.tif","twi.tif","vdepth.tif")
rstack <- stack(paste0("./DemLayers/",layers))
library(velox)
velStack <- velox(rstack)
trainDat <- st_read("TrainingData.gpkg")
testArea <- st_read(dsn = ".",layer = "TestArea3")
datSf <- st_transform(trainDat,3005)
datSub <- datSf[testArea,]
datSub <- st_transform(datSub,st_crs(rstack))
covDat <- raster::extract(rstack,datSub)
trDatComb <- data.table(ID = datSub$PlotNumber, Nutrient = datSub$NutrientRegime,covDat)
trDatComb[is.na(Aspect),Aspect := 0]
trDatComb <- na.omit(trDatComb)
fwrite(trDatComb,"TrainingData_Covariates.csv")

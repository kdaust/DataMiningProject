##Kiri Daust
##Script for compiling and creating layers for data mining project

library(data.table)
library(tidyverse)
library(sf)
library(raster)

##load plot data
dat <- fread("SoilTrainingData.csv")
dat <- dat[!is.na(Latitude),]
dat[,Longitude := -1*Longitude]
datSf <- st_as_sf(dat, coords = c("Longitude","Latitude"), crs = 4326)
datSf <- datSf[datSf$NutrientRegime %in% c("A","B","C","D","E"),]
st_write(datSf,dsn = "TrainingData.gpkg", driver = "GPKG")
plot(datSf["NutrientRegime"],pch = 16)
table(dat$NutrientRegime)

datSf <- st_read("TrainingData.gpkg")
testArea <- st_read(dsn = "./TestArea",layer = "FinalTA")
testArea <- st_transform(testArea,3005)
datSf <- st_transform(datSf,3005)
datSub <- datSf[testArea,]
datSub <- datSub[datSub$MoistureRegime %in% c(1,2,3,4,5,6),]
library(sen2r)


## create dem layers
library(Rsagacmd)
dem <- raster("bc25fill")
aoi <- st_read(dsn = "TestArea",layer = "FinalTA")
aoi <- st_transform(aoi,st_crs(dem))
dem2 <- crop(dem,aoi)
#crs(dem) <- 3001
dem2 <- st_as_stars(dem2)
dem3 <- st_warp(dem2, crs = 4326)
dem3 <- as(dem3,"Raster")
writeRaster(dem3, "DemClipped.asc", format = "ascii")
mat <- raster("./ClippedDem/Decade_2011_2019Y/MAT.asc")

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
smap_small <- smap[testArea,]
smap_small <- st_buffer(smap_small, dist = 0)
smap_small <- st_intersection(smap_small,testArea)
st_write(smap_small,dsn = "ClippedSoil.gpkg")
smap_small <- st_read("ClippedSoil.gpkg")
layerDat <- st_drop_geometry(smap_small)

##load dem for reference
tpi <- raster("./DemLayers/Aspect.tif")
library(fasterize)
snames <- c("SOILCODE_1","TEXTURE_1","Drain_1")
layNames <- c("Soilcode","Texture","Drainage")
soilStack <- stack()
for(lName in snames){
  temp <- smap_small[lName]
  colnames(temp)[1] <- "Var"
  temp$Var <- as.numeric(as.factor(temp$Var))
  temp <- st_cast(temp,"MULTIPOLYGON")
  rast <- fasterize(temp,raster = tpi,field = "Var")
  soilStack <- stack(soilStack,rast)
}
names(soilStack) <- layNames
bgcRast <- raster("BGCRaster.tif")
factorStack <- stack(soilStack,bgcRast)


library(stars)
ndvi <- raster("./SatelliteLayers/data_mining_ndvi2-0000000000-0000032768.tif")
st.ndvi <- st_as_stars(ndvi)
ref <- st_as_stars(tpi)
dvi2 <- st_warp(st.ndvi,dest = ref)
ndvi <- as(dvi2,"Raster")
writeRaster(ndvi,"ndvi.tif",format = "GTiff")

ndwi <- raster("./SatelliteLayers/data_mining_ndwi-0000000000-0000032768.tif")
st.ndwi <- st_as_stars(ndwi)
ref <- st_as_stars(tpi)
dwi2 <- st_warp(st.ndwi,dest = ref)
ndwi <- as(dwi2,"Raster")
writeRaster(ndwi,"ndwi.tif",format = "GTiff")

nbrt1 <- raster("./SatelliteLayers/data_mining_nbrt-0000000000-0000023296.tif")
nbrt2 <- raster("./SatelliteLayers/data_mining_nbrt-0000000000-0000046592.tif")
nbrt <- merge(nbrt1,nbrt2)
st.nbrt <- st_as_stars(nbrt)
brt2 <- st_warp(st.nbrt,dest = ref)
nbrt <- as(brt2,"Raster")
nbrt[nbrt > 1 | nbrt < -1] <- NA
writeRaster(nbrt,"nBRT.tif",format = "GTiff")

clay <- raster("./SatelliteLayers/data_mining_clay.tif")
st.clay <- st_as_stars(clay)
ref <- st_as_stars(tpi)
dwi2 <- st_warp(st.clay,dest = ref)
clay <- as(dwi2,"Raster")
writeRaster(clay,"clay.tif",format = "GTiff")


evi <- raster("./SatelliteLayers/data_mining_evi-0000000000-0000000000.tif")
evi2 <- raster("./SatelliteLayers/data_mining_evi-0000000000-0000023296.tif")
evi <- merge(evi1,evi2)
st.evi <- st_as_stars(evi)
evi2 <- st_warp(st.evi,dest = ref)
evi <- as(evi2,"Raster")
writeRaster(evi,"evi.tif",format = "GTiff")

iron <- raster("./SatelliteLayers/data_mining_iron-0000000000-0000000000.tif")
st.iron <- st_as_stars(iron)
iron2 <- st_warp(st.iron,dest = ref)
iron <- as(iron2,"Raster")
writeRaster(iron,"iron.tif",format = "GTiff")

rvi <- raster("./SatelliteLayers/data_mining_rvi-0000000000-0000000000.tif")
st.rvi <- st_as_stars(rvi)
rvi2 <- st_warp(st.rvi,dest = ref)
rvi <- as(rvi2,"Raster")
writeRaster(rvi,"rvi.tif",format = "GTiff")

bgc <- st_read("../Work2021/CommonTables/WNA_BGC_v12_12Oct2020.gpkg")
bgc <- st_buffer(bgc, dist = 0)
bgc <- st_intersection(bgc,testArea)
bgc <- bgc["BGC"]
bgcLookup <- data.table(BGC = unique(bgc$BGC))
bgcLookup[,bgcID := 1:nrow(bgcLookup)]
bgc <- merge(bgc,bgcLookup, by = "BGC", all = T)
bgc <- st_transform(bgc,st_crs(tpi))
bgc <- st_cast(bgc,"MULTIPOLYGON")
bgcRast <- fasterize(sf = bgc, raster = tpi,field = "bgcID")
writeRaster(bgcRast,"BGCRaster.tif",format = "GTiff")
##load satellite layers
slayers <- c("ndvi.tif","ndwi.tif","nBRT.tif","clay.tif","evi.tif","iron.tif") ##"evi.tif","rvi.tif",,"iron.tif"
satStack <- stack(paste0("./SatelliteLayers/",slayers))

##load layers, extract points
layers <- c("Aspect.tif","FlowAccum.tif","FlowLength.tif","mrrtf.tif","mrvbf.tif",
            "Slope.tif","tca.tif","tpi.tif","tri.tif","twi.tif","vdepth.tif")
rstack <- stack(paste0("./DemLayers/",layers))
allStack <- stack(rstack,satStack,factorStack)

# soilMap <- st_transform(smap_small,st_crs(allStack))
# samp <- st_sample(soilMap, size = 10000)
# samp <- st_as_sf(data.frame(ID = 1:10000,geometry = samp))
# soilSamp <- st_join(samp,soilMap,left = F)
# soilSamp <- unique(soilSamp)
# covDat <- raster::extract(allStack,samp)
# covDat <- data.table(ID = samp$ID,covDat)
# soilSamp <- st_drop_geometry(soilSamp)
# soilSamp <- as.data.table(soilSamp)
# soilSamp <- soilSamp[,.(ID,SOILCODE_1,PROFILE_1,PM1_1,Drain_1,TEXTURE_1)]
# covDat[soilSamp, Class := i.Drain_1, on = "ID"]
# covDat <- na.omit(covDat)
# covDat <- covDat[Class != "-",]
# table(covDat$Class)


trainDat <- st_read("TrainingData.gpkg")
datSf <- st_transform(trainDat,3005)
datSub <- datSf[testArea,]
datSub <- st_transform(datSub,st_crs(factorStack))
# factDat <- raster::extract(factorStack,datSub)
# factDat <- as.data.frame(apply(factDat,2,as.factor))
# factOHE <- model.matrix(~., data = factDat)
# factOHE <- factOHE[,-1]
allDat <- raster::extract(allStack,datSub)
allDat <- as.data.frame(allDat)
#allDat <- na.omit(allDat)
allDat$Soilcode <- as.factor(allDat$Soilcode); allDat$Drainage <- as.factor(allDat$Drainage)
allDat$Texture <- as.factor(allDat$Texture); allDat$BGCRaster <- as.factor(allDat$BGCRaster)
options(na.action = "na.pass")
allDatOHE <- model.matrix(~.,data = allDat)
allDatOHE <- allDatOHE[,-1]
options(na.action = "na.action.default")
allDatOHE <- allDatOHE[,abs(colSums(allDatOHE,na.rm = T)) > 5]

trDatComb <- data.table(Nutrient = datSub$NutrientRegime,allDatOHE)
trDatComb[is.na(Aspect),Aspect := 0]
trDatComb <- na.omit(trDatComb)
fwrite(trDatComb,"TrainingData_V2_OHE.csv")
trDatComb <- fread("TrainingData_V2_OHE.csv")
library(ranger)
trDatComb <- fread("TrainingData_NoNull.csv")
NutCross <- data.table(Nutrient = c("A","B","C","D","E"),NutrientOrd = c(0,0,0,1,1))
trDatComb[NutCross, YOrd := i.NutrientOrd, on = "Nutrient"]

#trDatComb[,YOrd := Nutrient]
trDatComb[,Nutrient := NULL]
trDatComb[,YOrd := as.factor(YOrd)]

library(UBL)
table(trDatComb$YOrd)
smoted <- SmoteClassif(YOrd ~ ., dat = trDatComb)
table(smoted$YOrd)
fwrite(smoted,"TrainingSet_Smoted.csv")
trDatComb <- smoted
set.seed(0)
# trDatComb[,ID := NULL]
# trDatComb <- trDatComb[Class %in% c("MW","R","W"),]
# trDatComb[,Class := as.factor(as.character(Class))]
testID <- sample(nrow(trDatComb),size = 0.7*nrow(trDatComb),replace = F)
trainDat <- trDatComb[testID,]
testDat <- trDatComb[-testID,]
library(ranger)
rf <- ranger(YOrd ~ .,data = trainDat,num.trees = 650,splitrule = "gini")
rf$prediction.error
temp <- predict(rf,testDat)
testDat$Fit <- temp$predictions
test <- testDat[,.(YOrd,Fit)]
table(test)
t2 <- test$YOrd == test$Fit
length(t2[t2  == T])/length(t2)
fwrite(trDatComb,"TrainingSmoted_2class.csv")
library(fastAdaboost)
ada <- adaboost(YOrd ~ ., data = trainDat,nIter = 100)
temp <- predict(ada,testDat)
testDat$Fit <- temp$class
test <- testDat[,.(YOrd,Fit)]
table(test)
t2 <- test$YOrd == test$Fit
length(t2[t2  == T])/length(t2)

library(ordinalForest)
trainDat[,YOrd := as.factor(YOrd)]
orf <- ordfor("YOrd",data = trainDat,perffunction = "proportional")


temp <- predict(orf,newdata = testDat)
testDat$Fit <- temp$ypred
test <- testDat[,.(YOrd,Fit)]
table(test)
t2 <- test$YOrd == test$Fit
length(t2[t2  == T])/length(t2)

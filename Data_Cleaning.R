library(data.table)
library(tidyverse)
library(sf)
library(raster)

dat <- fread("SoilTrainingData.csv")
dat <- dat[!is.na(Latitude),]
dat[,Longitude := -1*Longitude]
datSf <- st_as_sf(dat, coords = c("Longitude","Latitude"), crs = 4326)
plot(datSf["NutrientRegime"],pch = 16)
table(dat$NutrientRegime)

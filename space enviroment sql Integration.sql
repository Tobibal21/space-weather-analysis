DESCRIBE space_environment_dataset;

SELECT AVG(`Solar_Wind_Speed_km_s`)
FROM space_environment_dataset;

SELECT Day, Radiation_Level_mSv
FROM space_environment_dataset
WHERE Radiation_Level_mSv > 60;

SELECT COUNT(*)
FROM space_environment_dataset
WHERE Solar_Flare_Occurred = 1;
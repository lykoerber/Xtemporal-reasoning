# download TRAM benchmark
mkdir data
cd data

# ambiguity resolution
mkdir -p ambiguity_resolution
cd ambiguity_resolution
wget -c https://github.com/EternityYW/TRAM-Benchmark/raw/main/datasets/ambiguity_resolution.zip
unzip -q ambiguity_resolution.zip
rm ambiguity_resolution.zip
cd ..

# duration
mkdir duration
cd duration
wget -c https://github.com/EternityYW/TRAM-Benchmark/raw/main/datasets/duration.zip
unzip -q duration.zip
rm duration.zip
cd ..
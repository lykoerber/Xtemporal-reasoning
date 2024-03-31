# download TRAM benchmark
mkdir data
cd data

subcorpora=("ambiguity_resolution" \
            "arithmetic" \
            "causality" \
            "duration" \
            "frequency" \
            "nli_mcq" \
            "nli_saq" \
            "ordering" \
            "relation" \
            "storytelling" \
            "typical_time")

# Loop through subcorpora
for corpus in "${subcorpora[@]}"; do
    mkdir -p "$corpus"
    cd "$corpus" || exit
    wget -c "https://github.com/EternityYW/TRAM-Benchmark/raw/main/datasets/$corpus.zip"
    unzip -q "$corpus.zip"
    rm "$corpus.zip"
    cd ..
done

# merge nli saq and mcq datasets into one directory
mkdir nli
mv nli_mcq/* nli
mv nli_saq/* nli
rm -r nli_mcq
rm -r nli_saq

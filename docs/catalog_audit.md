# Public Catalog Audit

Audit date: 2026-07-22

Input: `data/public_shade_catalog.csv`

## Summary

- Total rows: 7,535
- Total brands: 106
- Total products: 318
- Exact duplicate `brand + product + shade_name + hex` rows: 0
- Sampled Top 3 duplicate check: 300 catalog-color probes produced no exact
  duplicate Top 3 recommendations.

The catalog appears to be foundation/base complexion focused after filtering.
It is not a manually curated brand catalog, so it still includes adjacent base
complexion products such as tinted moisturizers, BB/CC products, powder
foundations, foundation/concealer hybrids, and complexion sticks.

## Filtering Result

The importer now keeps rows with base-complexion signals such as foundation,
skin tint, tinted moisturizer, complexion, base, concealer, BB/CC, cushion,
cover drops, makeup, and teint/tint language.

It rejects obvious non-base categories such as lipstick, lip gloss, mascara,
eyeliner, eyeshadow, brow products, blush, bronzer, highlighter, fragrance, and
nail polish when there is no stronger complexion signal.

Post-filter importer summary:

- Total raw rows: 23,615
- Valid rows written: 7,535
- Skipped rows: 8,504
- Duplicate rows removed: 7,576
- Non-complexion rows skipped: 129
- Brands: 106
- Products: 318

`sephora.csv` and `ulta.csv` are skipped by the normalizer because they do not
include HEX swatch colors.

## Most Common Brands

```text
bareMinerals               407
MAKE UP FOR EVER           340
Tarte                      283
SEPHORA COLLECTION         253
Laura Mercier              252
Lancôme                    234
Clinique                   219
MAC                        185
KVD Vegan Beauty           176
COVER FX                   168
Too Faced                  161
Estée Lauder               160
Dior                       154
L'Oréal                    153
FENTY BEAUTY by Rihanna    150
```

## Most Common Products

```text
Ultra HD Invisible Cover Foundation                                100
Teint Idole Ultra Wear 24H Long Wear Foundation                     93
Lock-It Foundation                                                  85
Tinted Moisturizer Natural Skin Perfector Broad Spectrum SPF 30     80
Matte Velvet Skin Full Coverage Foundation                          80
#FauxFilter Skin Finish Buildable Coverage Foundation Stick         78
All Hours Longwear Natural Matte Foundation                         78
SEA Water Foundation Broad Spectrum SPF 15                          72
Foundation X+                                                       72
Studio Fix Fluid SPF 15 Foundation                                  63
Fluidity Full-Coverage Foundation                                   61
Matte Velvet Skin Blurring Powder Foundation                        60
10 Hour Wear Perfection Foundation                                  60
Tinted Moisturizer Broad Spectrum SPF 20 - Oil Free                 60
Synchro Skin Self-Refreshing Foundation SPF 30                      60
```

## Random 30-Row Sample

```csv
shade_id,brand,product,shade_name,hex,undertone,depth
PUBLIC-03650,Dior,Dior Airflash Spray Foundation,neutral,#A78466,neutral,tan
PUBLIC-01601,Dermablend,Smooth Liquid Camo Foundation,linen,#F6D7B1,unknown,light
PUBLIC-06743,MAC,Studio Fix Soft Matte Foundation Stick,NC16,#F1B789,warm,light-medium
PUBLIC-01468,Clinique,Even Better Makeup Broad Spectrum SPF 15,bone,#F1CFB3,unknown,light
PUBLIC-04054,Tarte,BB Tinted Treatment 12-Hour Primer Broad Spectrum SPF 30 Sunscreen,medium-tan,#D09B71,unknown,medium
PUBLIC-04088,Yves Saint Laurent,All Hours Longwear Natural Matte Foundation,warm almond,#DDAF8D,warm,light-medium
PUBLIC-07496,Givenchy,Teint Couture Everwear 24H Foundation SPF 20,Y207,#E6C29C,warm,light
PUBLIC-04588,Dior,Diorskin Forever Undercover Foundation,peach,#FEC6A0,warm,light
PUBLIC-00252,Lancôme,Teint Idole Ultra Longwear Foundation Stick SPF 21,suede,#AC7A49,unknown,tan
PUBLIC-01427,Revlon,ColorStay Makeup For Combo/Oily Skin,natural tan,#DCA787,neutral,light-medium
PUBLIC-03616,HUDA BEAUTY,#FauxFilter Skin Finish Buildable Coverage Foundation Stick,cinnamon,#D69D6D,unknown,medium
PUBLIC-06799,COVER FX,Pressed Mineral Foundation,N100,#A87355,neutral,tan
PUBLIC-04310,Bobbi Brown,Skin Foundation Stick,natural,#DCB086,neutral,light-medium
PUBLIC-05427,Armani Beauty,Neo Nude Foundation,4,#E9C3A4,unknown,light
PUBLIC-02489,Milani,Screen Queen Foundation,cool shell,#F8CDAD,cool,light
PUBLIC-01288,e.l.f. Cosmetics,Flawless Finish Foundation,tan,#DFAB78,unknown,light-medium
PUBLIC-04335,Shiseido,Synchro Skin Self-Refreshing Foundation SPF 30,alabaster,#F9E7CC,unknown,fair
PUBLIC-01047,Clinique,Beyond Perfecting Powder Foundation + Concealer,dune,#FBEED9,unknown,fair
PUBLIC-02465,ULTA,Effortless Effect Foundation,dark cool,#7B533D,cool,deep
PUBLIC-03721,SEPHORA COLLECTION,Matte Perfection Powder Foundation,warm toffee,#AF846B,warm,medium
PUBLIC-06123,Gucci,Fluide De Beauté Fini Naturel - Natural Finish Fluid Foundation,220N,#DBAD8D,neutral,light-medium
PUBLIC-06571,Origins,Pretty in Bloom™ SPF 20 Flower-Infused Long-Wear Foundation,520,#C4A088,unknown,light-medium
PUBLIC-03908,Bobbi Brown,Skin Long-Wear Weightless Foundation SPF 15,warm beige,#D5A685,warm,light-medium
PUBLIC-01731,Too Faced,Born This Way Undetectable Medium-to-Full Coverage Foundation,honey,#B17753,warm,tan
PUBLIC-06227,MAKE UP FOR EVER,Ultra HD Invisible Cover Foundation,R300,#C09C88,unknown,medium
PUBLIC-00971,Au Naturale,Zero Gravity C2P Foundation,lucerne,#D9B192,unknown,light-medium
PUBLIC-01035,Clinique,Almost Powder Makeup,fair,#F4F3F1,unknown,fair
PUBLIC-04866,SEPHORA COLLECTION,Matte Perfection Full Coverage Foundation,maple,#DBA280,unknown,light-medium
PUBLIC-03400,MAKE UP FOR EVER,Reboot Active Care Revitalizing Foundation,neutral beige,#CDA98A,neutral,light-medium
PUBLIC-01003,bareMinerals,BAREPRO Performance Wear Powder Foundation,sateen,#E6B295,unknown,light-medium
```

## Mixed-Category Check

No obvious lipstick, mascara, eyeliner, eyeshadow, brow product, bronzer,
highlighter, fragrance, or nail-polish product rows remain in the normalized
catalog.

The string `blush` appears in 8 remaining rows, but inspection shows those are
shade names within foundation or tinted-moisturizer products, for example
`PÜR 4-In-1 Foundation Stick - blush porcelain` and `Laura Mercier Tinted
Moisturizer - blush`. They do not appear to be blush product-category rows.

## Remaining Limitations

- The source is still a public website-derived swatch dataset, not measured
  physical foundation samples.
- Some base complexion products are adjacent to foundation rather than pure
  liquid foundation, including BB/CC creams, tinted moisturizers, powder
  foundations, and foundation/concealer hybrids.
- Product names are the main filtering signal. The normalized file does not
  preserve a detailed product category taxonomy for every source row.

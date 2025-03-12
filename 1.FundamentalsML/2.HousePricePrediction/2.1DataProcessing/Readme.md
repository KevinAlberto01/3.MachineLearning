<p align = "center" >
    <h1 align = "Center"> 2.1 Data Processing</h1>
</p>

This code performs data processing for a dataset predicting house prices (AmesHousing.csv). 
The workflow includes data loading, initial inspection, classification of columns into numerical and categorical, data cleaning (handling missing values), and standardization of column names. 
Finally, save the cleaned dataset in a CSV file.

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

**Data Loading:**
- Load the CSV file containing the house price dataset into a pandas DataFrame.

**Data Inspection:**
- Get an overview of the dataset before performing any processing, showing information about the data types and descriptive statistics.

**Column Classification:**
- Identify and classify the columns as numerical (float64, int64) and categorical (object).

**Data Cleaning:**
- Fill in the missing values in the numerical columns with the median of each column.
- Fill the missing values in the categorical columns with the value 'Missing'.
- Standardize the column names, removing spaces and converting everything to lowercase, with underscores between words.

**Save Cleaned Data:** 
- Save the processed and cleaned dataset in a CSV file for later use.

**Inspection After Cleaning:** 
- Check the dataset after cleaning, showing the information and descriptive statistics to ensure that missing values have been handled correctly.




<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<div style="margin-bottom: 1px; display: flex; justify-content: center; align-items: center; height: 100vh;">
    <details>
        <summary>📊 Dataset Information (Before Cleaning)</summary>
        <pre>
        <strong>#   Column           Non-Null Count  Dtype  </strong>
        --- ------           --------------  -----  
        0   Order            2930 non-null   int64  
        1   PID              2930 non-null   int64  
        2   MS SubClass      2930 non-null   int64  
        3   MS Zoning        2930 non-null   object 
        4   Lot Frontage     2440 non-null   float64
        5   Lot Area         2930 non-null   int64  
        6   Street           2930 non-null   object 
        7   Alley            198 non-null    object 
        8   Lot Shape        2930 non-null   object 
        9   Land Contour     2930 non-null   object 
        10  Utilities        2930 non-null   object 
        11  Lot Config       2930 non-null   object 
        12  Land Slope       2930 non-null   object 
        13  Neighborhood     2930 non-null   object 
        14  Condition 1      2930 non-null   object 
        15  Condition 2      2930 non-null   object 
        16  Bldg Type        2930 non-null   object 
        17  House Style      2930 non-null   object 
        18  Overall Qual     2930 non-null   int64  
        19  Overall Cond     2930 non-null   int64  
        20  Year Built       2930 non-null   int64  
        21  Year Remod/Add   2930 non-null   int64  
        22  Roof Style       2930 non-null   object 
        23  Roof Matl        2930 non-null   object 
        24  Exterior 1st     2930 non-null   object 
        25  Exterior 2nd     2930 non-null   object 
        26  Mas Vnr Type     1155 non-null   object 
        27  Mas Vnr Area     2907 non-null   float64
        28  Exter Qual       2930 non-null   object 
        29  Exter Cond       2930 non-null   object 
        30  Foundation       2930 non-null   object 
        31  Bsmt Qual        2850 non-null   object 
        32  Bsmt Cond        2850 non-null   object 
        33  Bsmt Exposure    2847 non-null   object 
        34  BsmtFin Type 1   2850 non-null   object 
        35  BsmtFin SF 1     2929 non-null   float64
        36  BsmtFin Type 2   2849 non-null   object 
        37  BsmtFin SF 2     2929 non-null   float64
        38  Bsmt Unf SF      2929 non-null   float64
        39  Total Bsmt SF    2929 non-null   float64
        40  Heating          2930 non-null   object 
        41  Heating QC       2930 non-null   object 
        42  Central Air      2930 non-null   object 
        43  Electrical       2929 non-null   object 
        44  1st Flr SF       2930 non-null   int64  
        45  2nd Flr SF       2930 non-null   int64  
        46  Low Qual Fin SF  2930 non-null   int64  
        47  Gr Liv Area      2930 non-null   int64  
        48  Bsmt Full Bath   2928 non-null   float64
        49  Bsmt Half Bath   2928 non-null   float64
        50  Full Bath        2930 non-null   int64  
        51  Half Bath        2930 non-null   int64  
        52  Bedroom AbvGr    2930 non-null   int64  
        53  Kitchen AbvGr    2930 non-null   int64  
        54  Kitchen Qual     2930 non-null   objec       
        55  TotRms AbvGrd    2930 non-null   int64  
        56  Functional       2930 non-null   object 
        57  Fireplaces       2930 non-null   int64  
        58  Fireplace Qu     1508 non-null   object 
        59  Garage Type      2773 non-null   object 
        60  Garage Yr Blt    2771 non-null   float64
        61  Garage Finish    2771 non-null   object 
        62  Garage Cars      2929 non-null   float64
        63  Garage Area      2929 non-null   float64
        64  Garage Qual      2771 non-null   object 
        65  Garage Cond      2771 non-null   object 
        66  Paved Drive      2930 non-null   object 
        67  Wood Deck SF     2930 non-null   int64  
        68  Open Porch SF    2930 non-null   int64  
        69  Enclosed Porch   2930 non-null   int64  
        70  3Ssn Porch       2930 non-null   int64  
        71  Screen Porch     2930 non-null   int64  
        72  Pool Area        2930 non-null   int64  
        73  Pool QC          13 non-null     object 
        74  Fence            572 non-null    object 
        75  Misc Feature     106 non-null    object 
        76  Misc Val         2930 non-null   int64  
        77  Mo Sold          2930 non-null   int64  
        78  Yr Sold          2930 non-null   int64  
        79  Sale Type        2930 non-null   object 
        80  Sale Condition   2930 non-null   object 
        81  SalePrice        2930 non-null   int64
        </pre>
    <details> 
</div>

<div style="margin-bottom: 1px; display: flex; justify-content: center; align-items: center; height: 100vh;">
    <details>
        <summary>📊 Summary Statistics (Before Cleaning)</summary>
        <pre>
        <strong>            Order           PID  MS SubClass  Lot Frontage       Lot Area  Overall Qual  Overall Cond   Year Built  Year Remod/Add  Mas Vnr Area  BsmtFin SF 1  BsmtFin SF 2  Bsmt Unf SF  Total Bsmt SF   1st Flr SF   2nd Flr SF  Low Qual Fin SF  Gr Liv Area  Bsmt Full Bath  Bsmt Half Bath    Full Bath    Half Bath  Bedroom AbvGr  Kitchen AbvGr  TotRms AbvGrd   Fireplaces  Garage Yr Blt  Garage Cars  Garage Area  Wood Deck SF  Open Porch SF  Enclosed Porch   3Ssn Porch  Screen Porch    Pool Area      Misc Val      Mo Sold      Yr Sold      SalePrice </strong>
        <strong>count</strong>   2930.00000  2.930000e+03  2930.000000   2440.000000    2930.000000   2930.000000   2930.000000  2930.000000     2930.000000   2907.000000   2929.000000   2929.000000  2929.000000    2929.000000  2930.000000  2930.000000      2930.000000  2930.000000     2928.000000     2928.000000  2930.000000  2930.000000    2930.000000    2930.000000    2930.000000  2930.000000    2771.000000  2929.000000  2929.000000   2930.000000    2930.000000     2930.000000  2930.000000   2930.000000  2930.000000   2930.000000  2930.000000  2930.000000    2930.000000
        <strong>mean</strong>   1465.50000  7.144645e+08    57.387372     69.224590   10147.921843      6.094881      5.563140  1971.356314     1984.266553    101.896801    442.629566     49.722431   559.262547    1051.614544  1159.557679   335.455973         4.676792  1499.690444        0.431352        0.061134     1.566553     0.379522       2.854266       1.044369       6.443003     0.599317    1978.132443     1.766815   472.819734     93.751877      47.533447       23.011604     2.592491     16.002048     2.243345     50.635154     6.216041  2007.790444  180796.060068
        <strong>std</strong>     845.96247  1.887308e+08    42.638025     23.365335    7880.017759      1.411026      1.111537    30.245361       20.860286    179.112611    455.590839    169.168476   439.494153     440.615067   391.890885   428.395715        46.310510   505.508887        0.524820        0.245254     0.552941     0.502629       0.827731       0.214076       1.572964     0.647921      25.528411     0.760566   215.046549    126.361562      67.483400       64.139059    25.141331     56.087370    35.597181    566.344288     2.714492     1.316613   79886.692357
        <strong>min</strong>      1.00000  5.263011e+08    20.000000     21.000000    1300.000000      1.000000      1.000000  1872.000000     1950.000000      0.000000      0.000000      0.000000     0.000000       0.000000   334.000000     0.000000         0.000000   334.000000        0.000000        0.000000     0.000000     0.000000       0.000000       0.000000       2.000000     0.000000    1895.000000     0.000000     0.000000      0.000000       0.000000        0.000000     0.000000      0.000000     0.000000      0.000000     1.000000  2006.000000   12789.000000
        <strong>25%</strong>   733.25000  5.284770e+08    20.000000     58.000000    7440.250000      5.000000      5.000000  1954.000000     1965.000000      0.000000      0.000000      0.000000   219.000000     793.000000   876.250000     0.000000         0.000000  1126.000000        0.000000        0.000000     1.000000     0.000000       2.000000       1.000000       5.000000     0.000000    1960.000000     1.000000   320.000000      0.000000       0.000000        0.000000     0.000000      0.000000     0.000000      0.000000     4.000000  2007.000000  129500.000000
        <strong>50%</strong>    1465.50000  5.354536e+08    50.000000     68.000000    9436.500000      6.000000      5.000000  1973.000000     1993.000000      0.000000    370.000000      0.000000   466.000000     990.000000  1084.000000     0.000000         0.000000  1442.000000        0.000000        0.000000     2.000000     0.000000       3.000000       1.000000       6.000000     1.000000    1979.000000     2.000000   480.000000      0.000000      27.000000        0.000000     0.000000      0.000000     0.000000      0.000000     6.000000  2008.000000  160000.000000
        <strong>75%</strong>    2197.75000  9.071811e+08    70.000000     80.000000   11555.250000      7.000000      6.000000  2001.000000     2004.000000    164.000000    734.000000      0.000000   802.000000    1302.000000  1384.000000   703.750000         0.000000  1742.750000        1.000000        0.000000     2.000000     1.000000       3.000000       1.000000       7.000000     1.000000    2002.000000     2.000000   576.000000    168.000000      70.000000        0.000000     0.000000      0.000000     0.000000      0.000000     8.000000  2009.000000  213500.000000
        <strong>max</strong>    2930.00000  1.007100e+09   190.000000    313.000000  215245.000000     10.000000      9.000000  2010.000000     2010.000000   1600.000000   5644.000000   1526.000000  2336.000000    6110.000000  5095.000000  2065.000000      1064.000000  5642.000000        3.000000        2.000000     4.000000     2.000000       8.000000       3.000000      15.000000     4.000000    2207.000000     5.000000  1488.000000   1424.000000     742.000000     1012.000000   508.000000    576.000000   800.000000  17000.000000    12.000000  2010.000000  755000.000000
        </pre>
    <details> 
</div>

<div style="margin-bottom: 1px; display: flex; justify-content: center; align-items: center; height: 100vh;">
    <details>
        <summary>📊 Dataset Information (After Cleaning)</summary>
        <pre>
        <strong>#  Column            Non-Null Count  Dtype</strong>  
        --- ------           --------------  -----  
        0   order            2930 non-null   int64  
        1   pid              2930 non-null   int64  
        2   ms_subclass      2930 non-null   int64  
        3   ms_zoning        2930 non-null   object 
        4   lot_frontage     2930 non-null   float64
        5   lot_area         2930 non-null   int64  
        6   street           2930 non-null   object 
        7   alley            2930 non-null   object 
        8   lot_shape        2930 non-null   object 
        9   land_contour     2930 non-null   object 
        10  utilities        2930 non-null   object 
        11  lot_config       2930 non-null   object 
        12  land_slope       2930 non-null   object 
        13  neighborhood     2930 non-null   object 
        14  condition_1      2930 non-null   object 
        15  condition_2      2930 non-null   object 
        16  bldg_type        2930 non-null   object 
        17  house_style      2930 non-null   object 
        18  overall_qual     2930 non-null   int64  
        19  overall_cond     2930 non-null   int64  
        20  year_built       2930 non-null   int64  
        21  year_remod/add   2930 non-null   int64  
        22  roof_style       2930 non-null   object 
        23  roof_matl        2930 non-null   object 
        24  exterior_1st     2930 non-null   object 
        25  exterior_2nd     2930 non-null   object 
        26  mas_vnr_type     2930 non-null   object 
        27  mas_vnr_area     2930 non-null   float64
        28  exter_qual       2930 non-null   object 
        29  exter_cond       2930 non-null   object 
        30  foundation       2930 non-null   object 
        31  bsmt_qual        2930 non-null   object 
        32  bsmt_cond        2930 non-null   object 
        33  bsmt_exposure    2930 non-null   object 
        34  bsmtfin_type_1   2930 non-null   object 
        35  bsmtfin_sf_1     2930 non-null   float64
        36  bsmtfin_type_2   2930 non-null   object 
        37  bsmtfin_sf_2     2930 non-null   float64
        38  bsmt_unf_sf      2930 non-null   float64
        39  total_bsmt_sf    2930 non-null   float64
        40  heating          2930 non-null   object 
        41  heating_qc       2930 non-null   object 
        42  central_air      2930 non-null   object 
        43  electrical       2930 non-null   object 
        44  1st_flr_sf       2930 non-null   int64  
        45  2nd_flr_sf       2930 non-null   int64  
        46  low_qual_fin_sf  2930 non-null   int64  
        47  gr_liv_area      2930 non-null   int64  
        48  bsmt_full_bath   2930 non-null   float64
        49  bsmt_half_bath   2930 non-null   float64
        50  full_bath        2930 non-null   int64  
        51  half_bath        2930 non-null   int64  
        52  bedroom_abvgr    2930 non-null   int64  
        53  kitchen_abvgr    2930 non-null   int64  
        54  kitchen_qual     2930 non-null   object 
        55  totrms_abvgrd    2930 non-null   int64  
        56  functional       2930 non-null   object 
        57  fireplaces       2930 non-null   int64  
        58  fireplace_qu     2930 non-null   object 
        59  garage_type      2930 non-null   object 
        60  garage_yr_blt    2930 non-null   float64
        61  garage_finish    2930 non-null   object 
        62  garage_cars      2930 non-null   float64
        63  garage_area      2930 non-null   float64
        64  garage_qual      2930 non-null   object 
        65  garage_cond      2930 non-null   object 
        66  paved_drive      2930 non-null   object 
        67  wood_deck_sf     2930 non-null   int64  
        68  open_porch_sf    2930 non-null   int64  
        69  enclosed_porch   2930 non-null   int64  
        70  3ssn_porch       2930 non-null   int64  
        71  screen_porch     2930 non-null   int64  
        72  pool_area        2930 non-null   int64  
        73  pool_qc          2930 non-null   object 
        74  fence            2930 non-null   object 
        75  misc_feature     2930 non-null   object 
        76  misc_val         2930 non-null   int64  
        77  mo_sold          2930 non-null   int64  
        78  yr_sold          2930 non-null   int64  
        79  sale_type        2930 non-null   object 
        80  sale_condition   2930 non-null   object 
        81  saleprice        2930 non-null   int64 
        </pre>
    <details> 
</div>

<div style="margin-bottom: 1px; display: flex; justify-content: center; align-items: center; height: 100vh;">
    <details>
        <summary>📊 Dataset Information (After Cleaning)</summary>
        <pre>
            order           pid  ms_subclass  lot_frontage       lot_area  overall_qual  overall_cond   year_built  year_remod/add  mas_vnr_area  bsmtfin_sf_1  bsmtfin_sf_2  bsmt_unf_sf  total_bsmt_sf   1st_flr_sf   2nd_flr_sf  low_qual_fin_sf  gr_liv_area  bsmt_full_bath  bsmt_half_bath    full_bath    half_bath  bedroom_abvgr  kitchen_abvgr  totrms_abvgrd   fireplaces  garage_yr_blt  garage_cars  garage_area  wood_deck_sf  open_porch_sf  enclosed_porch   3ssn_porch  screen_porch    pool_area      misc_val      mo_sold      yr_sold      saleprice
            count  2930.00000  2.930000e+03  2930.000000   2930.000000    2930.000000   2930.000000   2930.000000  2930.000000     2930.000000   2930.000000   2930.000000   2930.000000  2930.000000    2930.000000  2930.000000  2930.000000      2930.000000  2930.000000     2930.000000     2930.000000  2930.000000  2930.000000    2930.000000    2930.000000    2930.000000  2930.000000    2930.000000  2930.000000  2930.000000   2930.000000    2930.000000     2930.000000  2930.000000   2930.000000  2930.000000   2930.000000  2930.000000  2930.000000    2930.000000
            mean   1465.50000  7.144645e+08    57.387372     69.019795   10147.921843      6.094881      5.563140  1971.356314     1984.266553    101.096928    442.604778     49.705461   559.230717    1051.593515  1159.557679   335.455973         4.676792  1499.690444        0.431058        0.061092     1.566553     0.379522       2.854266       1.044369       6.443003     0.599317    1978.179522     1.766894   472.822184     93.751877      47.533447       23.011604     2.592491     16.002048     2.243345     50.635154     6.216041  2007.790444  180796.060068
            std     845.96247  1.887308e+08    42.638025     21.326422    7880.017759      1.411026      1.111537    30.245361       20.860286    178.634545    455.515036    169.142089   439.422500     440.541315   391.890885   428.395715        46.310510   505.508887        0.524762        0.245175     0.552941     0.502629       0.827731       0.214076       1.572964     0.647921      24.826620     0.760449   215.009876    126.361562      67.483400       64.139059    25.141331     56.087370    35.597181    566.344288     2.714492     1.316613   79886.692357
            min       1.00000  5.263011e+08    20.000000     21.000000    1300.000000      1.000000      1.000000  1872.000000     1950.000000      0.000000      0.000000      0.000000     0.000000       0.000000   334.000000     0.000000         0.000000   334.000000        0.000000        0.000000     0.000000     0.000000       0.000000       0.000000       2.000000     0.000000    1895.000000     0.000000     0.000000      0.000000       0.000000        0.000000     0.000000      0.000000     0.000000      0.000000     1.000000  2006.000000   12789.000000
            25%     733.25000  5.284770e+08    20.000000     60.000000    7440.250000      5.000000      5.000000  1954.000000     1965.000000      0.000000      0.000000      0.000000   219.000000     793.000000   876.250000     0.000000         0.000000  1126.000000        0.000000        0.000000     1.000000     0.000000       2.000000       1.000000       5.000000     0.000000    1962.000000     1.000000   320.000000      0.000000       0.000000        0.000000     0.000000      0.000000     0.000000      0.000000     4.000000  2007.000000  129500.000000
            50%    1465.50000  5.354536e+08    50.000000     68.000000    9436.500000      6.000000      5.000000  1973.000000     1993.000000      0.000000    370.000000      0.000000   466.000000     990.000000  1084.000000     0.000000         0.000000  1442.000000        0.000000        0.000000     2.000000     0.000000       3.000000       1.000000       6.000000     1.000000    1979.000000     2.000000   480.000000      0.000000      27.000000        0.000000     0.000000      0.000000     0.000000      0.000000     6.000000  2008.000000  160000.000000
            75%    2197.75000  9.071811e+08    70.000000     78.000000   11555.250000      7.000000      6.000000  2001.000000     2004.000000    162.750000    734.000000      0.000000   801.750000    1301.500000  1384.000000   703.750000         0.000000  1742.750000        1.000000        0.000000     2.000000     1.000000       3.000000       1.000000       7.000000     1.000000    2001.000000     2.000000   576.000000    168.000000      70.000000        0.000000     0.000000      0.000000     0.000000      0.000000     8.000000  2009.000000  213500.000000
            max    2930.00000  1.007100e+09   190.000000    313.000000  215245.000000     10.000000      9.000000  2010.000000     2010.000000   1600.000000   5644.000000   1526.000000  2336.000000    6110.000000  5095.000000  2065.000000      1064.000000  5642.000000        3.000000        2.000000     4.000000     2.000000       8.000000       3.000000      15.000000     4.000000    2207.000000     5.000000  1488.000000   1424.000000     742.000000     1012.000000   508.000000    576.000000   800.000000  17000.000000    12.000000  2010.000000  755000.000000
        </pre>
    <details> 
</div>



| Order | PID        | MS SubClass | MS Zoning | Lot Frontage | Lot Area | Street | Alley | Lot Shape | Land Contour | Utilities | Lot Config | Land Slope | Neighborhood | Condition 1 | Condition 2 | Bldg Type | House Style | Overall Qual | Overall Cond | Year Built | Year Remod/Add | Roof Style | Roof Matl | Exterior 1st | Exterior 2nd | Mas Vnr Type | Mas Vnr Area | Exter Qual | Exter Cond | Foundation | Bsmt Qual | Bsmt Cond | Bsmt Exposure | BsmtFin Type 1 | BsmtFin SF 1 | BsmtFin Type 2 | BsmtFin SF 2 | Bsmt Unf SF | Total Bsmt SF | Heating | Heating QC | Central Air | Electrical | 1st Flr SF | 2nd Flr SF | Low Qual Fin SF | Gr Liv Area | Bsmt Full Bath | Bsmt Half Bath | Full Bath | Half Bath | Bedroom AbvGr | Kitchen AbvGr | Kitchen Qual | TotRms AbvGrd | Functional | Fireplaces | Fireplace Qu | Garage Type | Garage Yr Blt | Garage Finish | Garage Cars | Garage Area | Garage Qual | Garage Cond | Paved Drive | Wood Deck SF | Open Porch SF | Enclosed Porch | 3Ssn Porch | Screen Porch | Pool Area | Pool QC | Fence | Misc Feature | Misc Val | Mo Sold | Yr Sold | Sale Type | Sale Condition | SalePrice |
|-------|------------|-------------|-----------|--------------|----------|--------|-------|-----------|--------------|-----------|------------|------------|--------------|-------------|-------------|-----------|-------------|--------------|--------------|------------|----------------|------------|-----------|--------------|--------------|--------------|--------------|------------|------------|------------|-----------|-----------|---------------|----------------|--------------|----------------|--------------|-------------|---------------|---------|------------|-------------|------------|------------|------------|-----------------|-------------|----------------|----------------|-----------|-----------|---------------|---------------|--------------|---------------|------------|------------|--------------|-------------|---------------|---------------|-------------|-------------|-------------|-------------|-------------|--------------|---------------|----------------|------------|--------------|-----------|---------|-------|--------------|----------|---------|---------|-----------|----------------|-----------|
|     1 | 0526301100 |         020 | RL        |          141 |    31770 | Pave   | NA    | IR1       | Lvl          | AllPub    | Corner     | Gtl        | NAmes        | Norm        | Norm        | 1Fam      | 1Story      |            6 |            5 |       1960 |           1960 | Hip        | CompShg   | BrkFace      | Plywood      | Stone        |          112 | TA         | TA         | CBlock     | TA        | Gd        | Gd            | BLQ            |          639 | Unf            |            0 |         441 |          1080 | GasA    | Fa         | Y           | SBrkr      |       1656 |          0 |               0 |        1656 |              1 |              0 |         1 |         0 |             3 |             1 | TA           |             7 | Typ        |          2 | Gd           | Attchd      |          1960 | Fin           |           2 |         528 | TA          | TA          | P           |          210 |            62 |              0 |          0 |            0 |         0 | NA      | NA    | NA           |        0 |       5 |    2010 | WD        | Normal         |    215000 |
|     2 | 0526350040 |         020 | RH        |           80 |    11622 | Pave   | NA    | Reg       | Lvl          | AllPub    | Inside     | Gtl        | NAmes        | Feedr       | Norm        | 1Fam      | 1Story      |            5 |            6 |       1961 |           1961 | Gable      | CompShg   | VinylSd      | VinylSd      | None         |            0 | TA         | TA         | CBlock     | TA        | TA        | No            | Rec            |          468 | LwQ            |          144 |         270 |           882 | GasA    | TA         | Y           | SBrkr      |        896 |          0 |               0 |         896 |              0 |              0 |         1 |         0 |             2 |             1 | TA           |             5 | Typ        |          0 | NA           | Attchd      |          1961 | Unf           |           1 |         730 | TA          | TA          | Y           |          140 |             0 |              0 |          0 |          120 |         0 | NA      | MnPrv | NA           |        0 |       6 |    2010 | WD        | Normal         |    105000 |
|     3 | 0526351010 |         020 | RL        |           81 |    14267 | Pave   | NA    | IR1       | Lvl          | AllPub    | Corner     | Gtl        | NAmes        | Norm        | Norm        | 1Fam      | 1Story      |            6 |            6 |       1958 |           1958 | Hip        | CompShg   | Wd Sdng      | Wd Sdng      | BrkFace      |          108 | TA         | TA         | CBlock     | TA        | TA        | No            | ALQ            |          923 | Unf            |            0 |         406 |          1329 | GasA    | TA         | Y           | SBrkr      |       1329 |          0 |               0 |        1329 |              0 |              0 |         1 |         1 |             3 |             1 | Gd           |             6 | Typ        |          0 | NA           | Attchd      |          1958 | Unf           |           1 |         312 | TA          | TA          | Y           |          393 |            36 |              0 |          0 |            0 |         0 | NA      | NA    | Gar2         |    12500 |       6 |    2010 | WD        | Normal         |    172000 |
|     4 | 0526353030 |         020 | RL        |           93 |    11160 | Pave   | NA    | Reg       | Lvl          | AllPub    | Corner     | Gtl        | NAmes        | Norm        | Norm        | 1Fam      | 1Story      |            7 |            5 |       1968 |           1968 | Hip        | CompShg   | BrkFace      | BrkFace      | None         |            0 | Gd         | TA         | CBlock     | TA        | TA        | No            | ALQ            |         1065 | Unf            |            0 |        1045 |          2110 | GasA    | Ex         | Y           | SBrkr      |       2110 |          0 |               0 |        2110 |              1 |              0 |         2 |         1 |             3 |             1 | Ex           |             8 | Typ        |          2 | TA           | Attchd      |          1968 | Fin           |           2 |         522 | TA          | TA          | Y           |            0 |             0 |              0 |          0 |            0 |         0 | NA      | NA    | NA           |        0 |       4 |    2010 | WD        | Normal         |    244000 |
|     5 | 0527105010 |         060 | RL        |           74 |    13830 | Pave   | NA    | IR1       | Lvl          | AllPub    | Inside     | Gtl        | Gilbert      | Norm        | Norm        | 1Fam      | 2Story      |            5 |            5 |       1997 |           1998 | Gable      | CompShg   | VinylSd      | VinylSd      | None         |            0 | TA         | TA         | PConc      | Gd        | TA        | No            | GLQ            |          791 | Unf            |            0 |         137 |           928 | GasA    | Gd         | Y           | SBrkr      |        928 |        701 |               0 |        1629 |              0 |              0 |         2 |         1 |             3 |             1 | TA           |             6 | Typ        |          1 | TA           | Attchd      |          1997 | Fin           |           2 |         482 | TA          | TA          | Y           |          212 |            34 |              0 |          0 |            0 |         0 | NA      | MnPrv | NA           |        0 |       3 |    2010 | WD        | Normal         |    189900 |


| order | pid        | ms_subclass | ms_zoning | lot_frontage | lot_area | street | alley   | lot_shape | land_contour | utilities | lot_config | land_slope | neighborhood | condition_1 | condition_2 | bldg_type | house_style | overall_qual | overall_cond | year_built | year_remod/add | roof_style | roof_matl | exterior_1st | exterior_2nd | mas_vnr_type | mas_vnr_area | exter_qual | exter_cond | foundation | bsmt_qual | bsmt_cond | bsmt_exposure | bsmtfin_type_1 | bsmtfin_sf_1 | bsmtfin_type_2 | bsmtfin_sf_2 | bsmt_unf_sf | total_bsmt_sf | heating | heating_qc | central_air | electrical | 1st_flr_sf | 2nd_flr_sf | low_qual_fin_sf | gr_liv_area | bsmt_full_bath | bsmt_half_bath | full_bath | half_bath | bedroom_abvgr | kitchen_abvgr | kitchen_qual | totrms_abvgrd | functional | fireplaces | fireplace_qu | garage_type | garage_yr_blt | garage_finish | garage_cars | garage_area | garage_qual | garage_cond | paved_drive | wood_deck_sf | open_porch_sf | enclosed_porch | 3ssn_porch | screen_porch | pool_area | pool_qc | fence   | misc_feature | misc_val | mo_sold | yr_sold | sale_type | sale_condition | saleprice |
|-------|------------|-------------|-----------|--------------|----------|--------|---------|-----------|--------------|-----------|------------|------------|--------------|-------------|-------------|-----------|-------------|--------------|--------------|------------|----------------|------------|-----------|--------------|--------------|--------------|--------------|------------|------------|------------|-----------|-----------|---------------|----------------|--------------|----------------|--------------|-------------|---------------|---------|------------|-------------|------------|------------|------------|-----------------|-------------|----------------|----------------|-----------|-----------|---------------|---------------|--------------|---------------|------------|------------|--------------|-------------|---------------|---------------|-------------|-------------|-------------|-------------|-------------|--------------|---------------|----------------|------------|--------------|-----------|---------|---------|--------------|----------|---------|---------|-----------|----------------|-----------|
|     1 |  526301100 |          20 | RL        |        141.0 |    31770 | Pave   | Missing | IR1       | Lvl          | AllPub    | Corner     | Gtl        | NAmes        | Norm        | Norm        | 1Fam      | 1Story      |            6 |            5 |       1960 |           1960 | Hip        | CompShg   | BrkFace      | Plywood      | Stone        |        112.0 | TA         | TA         | CBlock     | TA        | Gd        | Gd            | BLQ            |        639.0 | Unf            |          0.0 |       441.0 |        1080.0 | GasA    | Fa         | Y           | SBrkr      |       1656 |          0 |               0 |        1656 |            1.0 |            0.0 |         1 |         0 |             3 |             1 | TA           |             7 | Typ        |          2 | Gd           | Attchd      |        1960.0 | Fin           |         2.0 |       528.0 | TA          | TA          | P           |          210 |            62 |              0 |          0 |            0 |         0 | Missing | Missing | Missing      |        0 |       5 |    2010 | WD        | Normal         |    215000 |
|     2 |  526350040 |          20 | RH        |         80.0 |    11622 | Pave   | Missing | Reg       | Lvl          | AllPub    | Inside     | Gtl        | NAmes        | Feedr       | Norm        | 1Fam      | 1Story      |            5 |            6 |       1961 |           1961 | Gable      | CompShg   | VinylSd      | VinylSd      | Missing      |          0.0 | TA         | TA         | CBlock     | TA        | TA        | No            | Rec            |        468.0 | LwQ            |        144.0 |       270.0 |         882.0 | GasA    | TA         | Y           | SBrkr      |        896 |          0 |               0 |         896 |            0.0 |            0.0 |         1 |         0 |             2 |             1 | TA           |             5 | Typ        |          0 | Missing      | Attchd      |        1961.0 | Unf           |         1.0 |       730.0 | TA          | TA          | Y           |          140 |             0 |              0 |          0 |          120 |         0 | Missing | MnPrv   | Missing      |        0 |       6 |    2010 | WD        | Normal         |    105000 |
|     3 |  526351010 |          20 | RL        |         81.0 |    14267 | Pave   | Missing | IR1       | Lvl          | AllPub    | Corner     | Gtl        | NAmes        | Norm        | Norm        | 1Fam      | 1Story      |            6 |            6 |       1958 |           1958 | Hip        | CompShg   | Wd Sdng      | Wd Sdng      | BrkFace      |        108.0 | TA         | TA         | CBlock     | TA        | TA        | No            | ALQ            |        923.0 | Unf            |          0.0 |       406.0 |        1329.0 | GasA    | TA         | Y           | SBrkr      |       1329 |          0 |               0 |        1329 |            0.0 |            0.0 |         1 |         1 |             3 |             1 | Gd           |             6 | Typ        |          0 | Missing      | Attchd      |        1958.0 | Unf           |         1.0 |       312.0 | TA          | TA          | Y           |          393 |            36 |              0 |          0 |            0 |         0 | Missing | Missing | Gar2         |    12500 |       6 |    2010 | WD        | Normal         |    172000 |
|     4 |  526353030 |          20 | RL        |         93.0 |    11160 | Pave   | Missing | Reg       | Lvl          | AllPub    | Corner     | Gtl        | NAmes        | Norm        | Norm        | 1Fam      | 1Story      |            7 |            5 |       1968 |           1968 | Hip        | CompShg   | BrkFace      | BrkFace      | Missing      |          0.0 | Gd         | TA         | CBlock     | TA        | TA        | No            | ALQ            |       1065.0 | Unf            |          0.0 |      1045.0 |        2110.0 | GasA    | Ex         | Y           | SBrkr      |       2110 |          0 |               0 |        2110 |            1.0 |            0.0 |         2 |         1 |             3 |             1 | Ex           |             8 | Typ        |          2 | TA           | Attchd      |        1968.0 | Fin           |         2.0 |       522.0 | TA          | TA          | Y           |            0 |             0 |              0 |          0 |            0 |         0 | Missing | Missing | Missing      |        0 |       4 |    2010 | WD        | Normal         |    244000 |
|     5 |  527105010 |          60 | RL        |         74.0 |    13830 | Pave   | Missing | IR1       | Lvl          | AllPub    | Inside     | Gtl        | Gilbert      | Norm        | Norm        | 1Fam      | 2Story      |            5 |            5 |       1997 |           1998 | Gable      | CompShg   | VinylSd      | VinylSd      | Missing      |          0.0 | TA         | TA         | PConc      | Gd        | TA        | No            | GLQ            |        791.0 | Unf            |          0.0 |       137.0 |         928.0 | GasA    | Gd         | Y           | SBrkr      |        928 |        701 |               0 |        1629 |            0.0 |            0.0 |         2 |         1 |             3 |             1 | TA           |             6 | Typ        |          1 | TA           | Attchd      |        1997.0 | Fin           |         2.0 |       482.0 | TA          | TA          | Y           |          212 |            34 |              0 |          0 |            0 |         0 | Missing | MnPrv   | Missing      |        0 |       3 |    2010 | WD        | Normal         |    189900 |

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**NumPy:** Used to work with arrays and matrices, and it has functions for mathematics and statistics <br> **Pandas:** Used to manipulate and analyze data, particularly with DataFrames (tables of rows and columns).<br> **matplotlib.pyplot:** Used to create graphs, and pyplot is specifically for generating different types of plots (e.g., bar, line, scatter).<br> **load_digits from sklearn.datasets:** Used to load a dataset of handwritten digits, which is commonly used in classification projects.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/1.DeclareteLibraries.png" width="4000"/>|
**`digits`:** loads a dataset of handwritten digit images, where you find images of numbers 0 to 9 in black and white, each with 8x8 pixels. Each image represents a number (digit) and has a label indicating which number it is. <br> **`x`:** Contains images (numeric format), for each 8x8 image it is flattened into a 64 array, each value represents the pixel intensity (0 = black, 16 = white). <br> **`y`:** contains the labels (real numbers of the images), each element is a number from 0 to , representing which digit each image is| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/2.LoadDataset.png" width="4000"/>|
**`x.shape`:** Returns the dimension of x(1797, 64), it has 1797 images, and each image has 64 values (8x8 pixels). <br> **`y.shape`:** Returns the dimension of y (1797), there are 1797 labels per image, each label is the number it represents (0-9).|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/3.ExploringDimensionsDataSet.png" width="4000"/>|
**`clases`:** Contains the unique values of y(0-9) <br> **`count_classes`:** Array that indicates how many examples there are of each class | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/4.CheckClassesNumberExamples.png" width="4000"/>|
**`plt.figure(figsize=(8,5))`:** Set the size of the figure (8 wide and 5 high) <br> **`plt.bar(clases, count_classes, color='skyblue')`:** Create the bar chart, the list that contains the number of examples, assign the color of the bars. <br> **`plt.xlabel('Digit')`:** Set the x-axis label <br> **`plt.ylabel('Number of examples')`:** Set the y-axis label <br> **`plt.title('Distribution of the classes')`:** Add a title to the chart. <br> **`plt.show()`:** Visualize the graph | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/5.ViewDistributionClasses.png" width="4000"/>|
**`fig, axes = plt.subplots(2, 5, figsize=(10, 5))`:** Create a figure and a grid (2 rows and 5 columns) totaling 10 subplots, set the size (10, 5 in inches)  10 wide and 5 high <br> **`fig.suptitle("Examples of images")`:** Establish a general title <br> **`for i, ax in enumerate(axes.ravel())`:** Iterates through each of the subplots, converts the (2x5) matrix into a one-dimensional array, which makes individual access easier, and returns both the index `i` and the `ax` object in each iteration. <br> **`ax.imshow(x[i].reshape(8,8), cmap='gray')`:** it is a vector image from the dataset.  resize to an 8x8 matrix, apply a grayscale <br> **`ax.set_title(f"label: {y[i]}")`:** Assign a title to each subplot. <br> **`ax.axis('off')`:** Deactivate the axes so that the marks or values do not appear. <br> **`plt.show()`:** Show the subplots and images. | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/6.ViewExample.png" width="4000"/>|

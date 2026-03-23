# molecular-id-marker-selection
Ultra-fast molecular identity marker selection tool optimized for large-scale genomic datasets. This tool screens SNP/SSR markers to ensure sufficient genetic differences between samples, supports distance constraints, parallel computing, and outputs similar sample groups for molecular ID construction in genetics/breeding research.

Usage:
  python3 marker.py -i <input> -o <output> -s <similar-output> [options] 
 
Required Arguments:

  -i, --input           Input marker file (form the "plink --bfile bfile --recode A-transpose --out data --allow-extra-chr")
 
  -o, --output          Output file for selected markers (required)
  
  -s, --similar-output  Output file for similar sample groups (required)

Optional Arguments:

  -k, --min-differences INT   Minimum differences required between sample pairs [default: 3]
  
  -d, --distance INT          Minimum distance between markers (bp) [default: 0]
  
  -b, --batch-size INT        Batch size for processing markers [default: 50000]

  --nochr                 Disable chromosome-balanced selection
      
  --parallel               Enable multi-process parallel computing
      
  --workers INT            Number of parallel workers [default: 4]
      
  --max-markers INT        Maximum number of markers to select [default: 1000]
      
  --min-saturation FLOAT   Stop when saturation reaches this value [default: 0.95]
      
  --strategy STRATEGY      Selection strategy: fast / balanced / accurate [default: balanced]
      
  -h, --help                  Show this help message and exit
  

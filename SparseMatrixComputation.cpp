/* Implementation of Sparse Matrix Computation in parHDE
** inut parameters: 
**  distMatrix (n x numSources) - matrix containing the distances
**  degrees (n) - degrees of the vertices
**  g - input graph with rowOffsets, adj as members
**  numSources - number of sources the distance were calculated from
**  n - number of vertices in the graph
*/
double * parHDE_SparseMatrixComp(double *distMatrix, double *degrees, graph_t *g, int numSources, int n)
{
    double *LXMatrix; 
    LXMatrix = (double *)mkl_malloc( n * numSources * sizeof( double ), 64 );

    /* Performaing vector-vector multiplication of "numSources" vectors */
    #pragma omp parallel for collapse(2) schedule(guided)
    for(unsigned k=0; k < numSources; k++)
    {
        for(long i=0; i < n; i++)
        {
            LXMatrix[i + k*n] = degrees[i] * distMatrix[i + k*n];
            for(unsigned int j = g->rowOffsets[i]; j < g->rowOffsets[i+1]; j++)
            {
                unsigned int v = g->adj[j];
                LXMatrix[i + k*n] -= distMatrix[v + k*n];
            }
        }
    }
    return LXMatrix;
}

/* Implementation of Sparse Matrix Computation with MKL routines
** inut parameters: 
**  distMatrix (n x numSources) - matrix containing the distances
**  degrees (n) - degrees of the vertices
**  g - input graph with rowOffsets, adj as members
**  numSources - number of sources the distance were calculated from
**  n - number of vertices in the graph
**  m - number of non zeros in the adjacency matrix
*/
double * mkl_SparseMatrixComp(double *distMatrix, double *degrees, graph_t *g, int numSources, int n, int m)
{
    /* MKL Sparse data structure initialization */
    MKL_INT *rows_start, *rows_end, *col_indx;
    double *values;
    rows_start = (MKL_INT *) mkl_malloc (  n   * sizeof(MKL_INT), 64);
    rows_end   = (MKL_INT *) mkl_malloc (  n   * sizeof(MKL_INT), 64);
    col_indx   = (MKL_INT *) mkl_malloc ((m+n) * sizeof(MKL_INT), 64);
    values     = (double  *) mkl_malloc ((m+n) * sizeof(double) , 64);
    sparse_matrix_t csrA = NULL;
    sparse_index_base_t indexing;
    struct matrix_descr descr_type_gen;
    descr_type_gen.type = SPARSE_MATRIX_TYPE_GENERAL;
    rows_start[0] = 0;
    MKL_INT edge_count = 0, temp_indx, status=0;
    double temp_value;	
    for(long i=0; i < n; i++) 
    {	
        rows_start[i] = edge_count;
        unsigned v;
        for(unsigned j=g->rowOffsets[i]; j < g->rowOffsets[i+1]; j++)
        {
            v = g->adj[j];
            assert(i != v);
            // insert for (i,i) in order
            if((j == g->rowOffsets[i] && i < v) || (j != g->rowOffsets[i] && i < v && i > g->adj[j-1]))
            {
                col_indx[edge_count] = i;
                values[edge_count] = degrees[i];
                edge_count++;
            }
            // insertion for (i,j)
            col_indx[edge_count] = (MKL_INT) v;
            values[edge_count] = -1.0; 
            edge_count++;
        }
        // if all j's < i
        if( i > v) 
        {
            col_indx[edge_count] = i;
            values[edge_count] = degrees[i];
            edge_count++;
        }
    }
    for( long i=0; i < n-1; i++)
        rows_end[i] = rows_start[i+1];
    assert( edge_count == m+n);
    rows_end[n-1] = edge_count;
    /* Creation of sparse_matrix_t struct */
    mkl_sparse_d_create_csr ( &csrA, 
                              SPARSE_INDEX_BASE_ZERO,
                              n,  // number of rows
                              n,  // number of cols
                              rows_start,
                              rows_end,
                              col_indx,
                              values );

    /* Sparse matrix computation */
    double *LXMatrix; 
    LXMatrix = (double *)mkl_malloc( n * numSources * sizeof( double ), 64 );
    mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1,
                     csrA, descr_type_gen, SPARSE_LAYOUT_COLUMN_MAJOR, distMatrix,
					 numSources, n, 0, LXMatrix, n );
    return LXMatrix;
}
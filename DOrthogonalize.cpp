/* Implementation of Dortho in parHDE  
** inut parameters: 
**  distMatrix (n x numSources) - matrix containing the distances
**  degrees (n) - degrees of the vertices
**  numSources - number of sources the distance were calculated from
**  n - number of vertices in the graph
*/
void parHDE_DOrthogonalize(double *distMatrix, double *degrees, int numSources, int n) 
{
  int j = 1;
  for (int run_count=0; run_count < numSources; run_count++) 
  {
    for (int k=0; k<j; k++) 
    {
        double multplr_denom=0, multplr_num=0;
        #pragma omp parallel
        {
            #pragma omp for simd reduction(+: multplr_denom, multplr_num)
            for(long p=0; p < n; p++)
            {
                double dnorm = distMatrix[p + k*n] * degrees[p]; // degree normalized vector
                // Dorthogonalize multipliers
                multplr_denom += distMatrix[p + k*n] * dnorm;
                multplr_num += distMatrix[p + j*n] * dnorm;
            }
            #pragma omp single
                multplr_denom = 1 / multplr_denom;
            #pragma omp for simd
            for(long p=0; p < n; p++) // Dorthogonalization
                distMatrix[p + j*n]  -= (multplr_num * distMatrix[p + k*n] * multplr_denom);
        }
    }

    double normdist = cblas_dnrm2(n, distMatrix + j*n, 1);
    if (normdist < 0.001) 
    {
      std::cout << "discarding vec " << j << ", normdist " << normdist << std::endl;
      j--;
    } 
    else 
    {
        normdist = 1 / normdist;
        #pragma omp parallel for simd
        for(long p=0; p < n; p++)
        distMatrix[p + j*n] *= normdist;
    }

    #pragma omp parallel for simd
    for(long p=0; p < n; p++)
	    distMatrix[p + (j-1)*n] = distMatrix[p + j*n];
    j++;

  }

}

/* Implemenation of Dortho in parHDE with MKL routines
** inut parameters: 
**  distMatrix (n x numSources) - matrix containing the distances
**  degrees (n) - degrees of the vertices
**  numSources - number of sources the distance were calculated from
**  n - number of vertices in the graph
*/
void mkl_Dorthogonalize(double *distMatrix, double *degrees, int numSources, int n) 
{
    double *dnormvec = (double *)mkl_malloc( n*sizeof(double), 64);
    int j = 1;
    for (int run_count=0; run_count < numSources; run_count++) 
    {
        for (int k=0; k<j; k++) 
        {
            #pragma omp parallel for simd 
            for(long p=0; p < n; p++) // degree normalized vector
                dnormvec[p] = distMatrix[p + k * n] * degrees[p];
            // cblas dot product for Dorthogonalize multipliers
            double multplr_denom =  cblas_ddot(n, (distMatrix + k * n), 1, dnormvec, 1);
            double multplr_num = cblas_ddot(n, (distMatrix + j * n), 1, dnormvec, 1);
            multplr_denom = 1 / multplr_denom;
            // cblas vector-vector update for dorthogonalization
            cblas_daxpy(n, -(multplr_num * multplr_denom), (distMatrix + k*n), 1, (distMatrix + j*n), 1);
        }

        // cblas eucleadian norm
        double normdist = cblas_dnrm2(n, (distMatrix + j * n), 1);
        if (normdist < 0.001)
        {
            std::cout << "discarding vec " << j << ", normdist " << normdist << std::endl;
            j--;
        }
        else
        {
            normdist = 1 / normdist;
            // cblas scalar update
            cblas_dscal(n, normdist, (distMatrix + j * n), 1);
        }
        // cblas vectory copy
        cblas_dcopy(n, (distMatrix + j * n), 1, (distMatrix + (j - 1) * n), 1);
        j++;
    }
    mkl_free(dnormvec);
}
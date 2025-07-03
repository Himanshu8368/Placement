#include<vector>
using namespace std;
// Memoization approach
// Time complexity: O(m*n)
class Solution {
public:
    int solve(int i,int j,vector<vector<int>>& dp){
        if(i<0 ||j<0) return 0;
        if(i==0 && j==0) return 1;
        if(dp[i][j]!=-1) return dp[i][j];

        int up_path =  solve(i-1,j,dp);
        int left_path =solve(i,j-1,dp);
        return dp[i][j] = up_path +left_path;

    }
    int uniquePaths(int m, int n) {
        vector<vector<int> >dp(m,vector<int>(n,-1));
        return solve(m-1,n-1,dp);
        // return dp[m-1][n-1];
    }
};
// Tabulation approach
// Time complexity: O(m*n)
int solve(int m,int n,vector<vector<int>>& dp){
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(i==0 ||j==0){
                    dp[i][j]=1;
                    continue;
                }
                int up =0;
                int left =0;
                if(i>0) up = dp[i-1][j];
                if(j>0) left = dp[i][j-1];
                dp[i][j]= up+ left;
            }
        }
        return dp[m-1][n-1];

    }
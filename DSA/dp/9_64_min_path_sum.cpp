#include <vector>
#include <limits.h>
using namespace std;
// Memoization approach
// Time complexity: O(m*n)
class Solution {
public:
    int solve(int i,int j ,vector<vector<int>>& grid, vector<vector<int>>& dp){
        if(i<0 || j<0){
            return INT_MAX;
        }
        if(i==0 && j==0){
            return grid[i][j];
        }
        if(dp[i][j]!=-1) return dp[i][j];
        int up = solve(i-1,j,grid,dp);
        int left = solve(i,j-1,grid,dp);

        return dp[i][j] = grid[i][j]+ min(up,left);
    }
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<int>> dp(m,vector<int>(n,-1));
        return solve(m-1,n-1,grid,dp);
    }
};

// Tabulation approach
// Time complexity: O(m*n)
class Solution1 {
public:
    // int solve(int i,int j ,vector<vector<int>>& grid, vector<vector<int>>& dp){
    //     if(i<0 || j<0){
    //         return INT_MAX;
    //     }
    //     if(i==0 && j==0){
    //         return grid[i][j];
    //     }
    //     if(dp[i][j]!=-1) return dp[i][j];
    //     int up = solve(i-1,j,grid,dp);
    //     int left = solve(i,j-1,grid,dp);

    //     return dp[i][j] = grid[i][j]+ min(up,left);
    // }
    int solve(int m,int n ,vector<vector<int>>& grid, vector<vector<int>>& dp){
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(i==0 && j==0){
                    dp[i][j]=grid[i][j];
                    continue;
                }
                int up =INT_MAX;
                int left =INT_MAX;
                if(i>0) up = dp[i-1][j];
                if(j>0) left = dp[i][j-1];
                dp[i][j] = grid[i][j] + min(up,left);
            }
        }
        return dp[m-1][n-1];
    }
    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        vector<vector<int>> dp(m,vector<int>(n,-1));
        return solve(m,n,grid,dp);
    }
};







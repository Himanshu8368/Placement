#include <vector>
using namespace std;
// This code is an optimized version of searching in a 2D matrix where each row and column is sorted.

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int n = matrix.size();
        int m = matrix[0].size();
        int col = m-1;
        int row =0;
        while(row<n && col>=0){
            if(matrix[row][col]==target) return true;
            if(matrix[row][col]>target) col--;
            else{
                row++;
            }
        }
        return false;
    }
};
// complexity: O(n+m) where n is the number of rows and m is the number of columns in the matrix.
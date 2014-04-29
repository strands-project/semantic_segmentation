/**
 * @file   console_tree.hh
 * @author Alexander Hermans
 * @date   Thu Nov 15 15:47:00 2012
 *
 * @brief  Class printing out visualizations during tree training
 *
 *
 */

#ifndef _UTILS_CONSOLE_TREE_H_
#define _UTILS_CONSOLE_TREE_H_

// STL-Headers
#include <vector>


#define SCREEN_WIDTH_IN_CHARS      230
#define MAX_VISUALIZED_TREE_DEPTH  6



namespace Utils {
  class Console_Tree {
  public:
    Console_Tree();
    ~Console_Tree();

    void Init(int trained_tree_depth_max_depth, int screen_width=SCREEN_WIDTH_IN_CHARS , int depth_skip=MAX_VISUALIZED_TREE_DEPTH);

    void PrintRoot();
    void PrintLeftChild(int depth);
    void PrintRightChild(int depth, int status);
    void Update(int percent, int second_info);
    void Update(int percent);
    void PopBack(int depth, int status);

  private:
    int m_screen_width;
    int m_depth_skip;
    int m_max_depth;
    int m_trained_tree_max_depth;

    int m_test;

    std::vector<int> m_current_col;
    std::vector<int> m_current_row;
    std::vector<int> m_current_width;
    std::vector<int> m_current_progress;
    std::vector<bool> m_current_is_leaf;

    void cls();
    void pos(int row, int col);

    void PrintEdge(int row_parent, int col_parent, int row_child, int col_child, bool left);
    void PrintNode(int row, int col, int percent);
    void PrintLeaf(int row, int col, int status);

  };
}




#endif /* _UTILS_CONSOLE_TREE_H_ */

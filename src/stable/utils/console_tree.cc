/**
 * @file   console_tree.cc
 * @author Alexander Hermans
 * @date   Thu Nov 15 15:47:00 2012
 *
 * @brief  Class printing out visualizations during tree training
 *
 */

// Local includes
#include "console_tree.hh"

// STL-Headers
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace Utils;

Console_Tree::Console_Tree(){

}

Console_Tree::~Console_Tree(){

}

void Console_Tree::Init(int trained_tree_depth_max_depth,int screen_width, int depth_skip){
  cls(); //Clear the screen.
  pos(0,0);
  printf(" \033[46m                              \033[0m\n");
  printf(" \033[46m \033[33m _______\033[35m\\)%%%%%%%%%%%%%%%%._\033[0m\033[46m         \033[0m\n");
  printf(" \033[46m \033[33m`''''-'\033[0m\033[46m-;   \033[35m%% %% %% %% %%\033[0m\033[46m'-._    \033[0m\n");
  printf(" \033[46m         :\033[31mO\033[0m\033[46m) \\           '-.  \033[0m\n");
  printf(" \033[46m         : :__)'    .'    .'  \033[0m\n");
  printf(" \033[46m         :.::/  '.'   .'      \033[0m\n");
  printf(" \033[46m         o_j/   :    ;        \033[0m\n");
  printf(" \033[46m                :   .'        \033[0m\n");
  printf(" \033[46m                 ''`          \033[0m\n");
  printf("\n  Leaf node colors: \n");
  printf("  \033[44m \033[0m  Too few samples to split\n\n");
  printf("  \033[41m \033[0m  No split could be found\n\n");
  printf("  \033[42m \033[0m  Pure class node\n\n");
  printf("  \033[43m \033[0m  Should never happen!\n\n");


  m_trained_tree_max_depth = trained_tree_depth_max_depth;
  m_screen_width = screen_width;
  m_depth_skip = depth_skip;
  m_max_depth = 0;
  m_current_is_leaf.push_back(true);
  while(pow(2,m_max_depth)*7 <= m_screen_width ){
    m_max_depth++;
  }
}

void Console_Tree::PrintRoot(){
  m_current_col.push_back(m_screen_width/2);
  m_current_row.push_back(3);
  m_current_width.push_back(m_screen_width);
  m_current_progress.push_back(0);
  PrintNode(m_current_row.back(), m_current_col.back(),0);
}

void Console_Tree::PrintLeftChild(int depth){
  m_current_is_leaf.back()=false;

  int d = m_trained_tree_max_depth-depth;
  if(d==m_max_depth){
    PrintEdge(m_current_row.back(), m_current_col.back(), m_current_row.back()+5, m_current_col.back(), false);
    pos(m_current_row.back()+5, m_current_col.back()+1);
    printf("\033[44m...\033[0m");
    return;
  }else if(d<m_max_depth){
    //Print the new node
    int row_left, col_left, width_new;
    row_left = m_current_row.back()+3+m_depth_skip;
    col_left = m_current_col.back() - (m_current_width.back()/4);
    width_new = m_current_width.back()/2;
    PrintNode(row_left, col_left, 0);

    //Print the new edge
    PrintEdge(m_current_row.back(), m_current_col.back(), row_left, col_left, true);


    m_current_col.push_back(col_left);
    m_current_row.push_back(row_left);
    m_current_width.push_back(width_new);
    m_current_progress.push_back(0);
    m_current_is_leaf.push_back(true);
  }

}

void Console_Tree::PrintRightChild(int depth, int status){
  int d = m_trained_tree_max_depth-depth;
  if(d==m_max_depth){
    PrintEdge(m_current_row.back(), m_current_col.back(), m_current_row.back()+5, m_current_col.back(), false);
    pos(m_current_row.back()+5, m_current_col.back()+1);
    printf("\033[44m...\033[0m");
    return;
  }else if(d<m_max_depth){
    PopBack(depth, status);

    //Print the new node
    int row_right, col_right, width_new;
    row_right = m_current_row.back()+3+m_depth_skip;
    col_right = m_current_col.back() + (m_current_width.back()/4);
    width_new = m_current_width.back()/2;
    PrintNode(row_right, col_right, 0);

    //Print the new edge
    PrintEdge(m_current_row.back(), m_current_col.back(), row_right, col_right, false);


    m_current_col.push_back(col_right);
    m_current_row.push_back(row_right);
    m_current_width.push_back(width_new);
    m_current_progress.push_back(0);
    m_current_is_leaf.push_back(true);
  }
}

void Console_Tree::PopBack(int depth, int status){
  if((m_trained_tree_max_depth-depth)<m_max_depth){
    if(m_current_is_leaf.back()){
      //We have a leave node here. Draw it in blue!
      PrintLeaf(m_current_row.back(), m_current_col.back(), status);
    }

    m_current_col.pop_back();
    m_current_row.pop_back();
    m_current_width.pop_back();
    m_current_progress.pop_back();
    m_current_is_leaf.back()=false;
  }
  if((m_trained_tree_max_depth-depth)==1){
    //We are done. Reset to a nice screen position
    pos(200,0);
  }

}

void Console_Tree::Update(int percent, int second_info){
  if(percent > m_current_progress.back()){
    m_current_progress.back() = percent;
      PrintNode(m_current_row.back(), m_current_col.back(), second_info);
  }
}

void Console_Tree::Update(int percent){
  if(percent > m_current_progress.back()){
    m_current_progress.back() = percent;
    PrintNode(m_current_row.back(), m_current_col.back(), percent);
  }
}




void Console_Tree::cls(){
  printf("%c[2J",27);
}

void Console_Tree::pos(int row, int col){
  printf("%c[%d;%dH",27,row,col);
}

void Console_Tree::PrintEdge(int row_parent, int col_parent, int row_child, int col_child, bool left){
  // Params are top left corners, we need to add some variables.
  int rP = row_parent+3;
  int cP = col_parent+2;
  int rC = row_child-1;
  int cC = col_child+2;
  int col_offset;
  if(left){
    col_offset = cP - cC;
  }else{
    col_offset = cC - cP;
  }
  float row_offset = static_cast<float>(rC- rP)/static_cast<float>(col_offset);
  float r;
  int c;
  r=rP;
  c=cP;
  if(left){
    for( ;  c>=cC; c--, r+=row_offset){
      pos(static_cast<int>(r+0.5),c);
      printf("*");
    }
  }else{
    for( ;  c<=cC; c++, r+=row_offset){
      pos(static_cast<int>(r+0.5),c);
      printf("*");
    }
  }
}

void Console_Tree::PrintLeaf(int row, int col, int status){
  std::string color;
  if(status == 1){ //Too few samples => blue
    color = " \033[44m   \033[0m ";
  }else if(status == 2){ //No split! => red
    color = " \033[41m   \033[0m ";
  }else if(status == 3){ //Pure node => green
    color = " \033[42m   \033[0m ";
  }else{ //Mistake => yellow ?
    color = " \033[43m   \033[0m ";
  }
  pos(row,col);
  printf(color.data());
  pos(row+1,col);
  printf(color.data());
  pos(row+2,col);
  printf("     ");
  fflush(stdout);
}


void Console_Tree::PrintNode(int row, int col, int percent){
  pos(row,col);
  printf("\033[47m     \033[0m");
  pos(row+1,col);
  printf("\033[47m \033[0m%3d\033[47m \033[0m", percent);
  pos(row+2,col);
  printf("\033[47m     \033[0m");
  fflush(stdout);
}

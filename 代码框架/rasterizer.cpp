// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include <unordered_map>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    //赋予这一组(两个)三角形一个编号
    auto id = get_next_id();

    //记录这个编号对应的一组三角形
    pos_buf.emplace(id, positions);

    //返回编号
    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    //同上
    //把这组三角形每个顶点的编号记录下来
    //并赋予一个总的编号
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    //同上,记录颜色
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(int x, int y, const Vector3f* _v, 
Vector3f side01 ,Vector3f side12 ,Vector3f side20)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]

    Vector3f side0 = _v[0] - ( VectorXf(3) << 1.0*x , 1.0*y , 0.0 ).finished();
    Vector3f side1 = _v[1] - ( VectorXf(3) << 1.0*x , 1.0*y , 0.0 ).finished();
    Vector3f side2 = _v[2] - ( VectorXf(3) << 1.0*x , 1.0*y , 0.0 ).finished();

    Vector3f cross0 = side01.cross(side1); 
    Vector3f cross1  =side12.cross(side2); 
    Vector3f cross2 = side20.cross(side0); 

    if(cross0[2] > 0 &&cross1[2] > 0 &&cross2[2] > 0)
        return true;
    if(cross0[2] < 0 &&cross1[2] < 0 &&cross2[2] < 0)
        return true;
    
    return false;

}

static bool insideTriangle(float x, float y, const Vector3f* _v, 
Vector3f side01 ,Vector3f side12 ,Vector3f side20)
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]

    Vector3f side0 = _v[0] - ( VectorXf(3) << 1.0*x , 1.0*y , 0.0 ).finished();
    Vector3f side1 = _v[1] - ( VectorXf(3) << 1.0*x , 1.0*y , 0.0 ).finished();
    Vector3f side2 = _v[2] - ( VectorXf(3) << 1.0*x , 1.0*y , 0.0 ).finished();

    Vector3f cross0 = side01.cross(side1); 
    Vector3f cross1  =side12.cross(side2); 
    Vector3f cross2 = side20.cross(side0); 

    if(cross0[2] > 0 &&cross1[2] > 0 &&cross2[2] > 0)
        return true;
    if(cross0[2] < 0 &&cross1[2] < 0 &&cross2[2] < 0)
        return true;
    
    return false;

}



static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    //提取出这组三角形的顶点,编号,颜色
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    //0.1 zNear
    //50 zFar
    float f1 = (50 - 0.1) / 2.0;
    float f2 = (-50 - 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    
    //遍历每个三角形
    for (auto& i : ind)
    {
        //把该三角形的每个点提取出来
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };

        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }

        //把[-1,1]^3放大回原来大小
        //并沿着z把图形移回去
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
        }

        auto col_0 = col[i[0]];
        auto col_1 = col[i[1]];
        auto col_2 = col[i[2]];

        t.setColor(0, col_0[0], col_0[1], col_0[2]);
        t.setColor(1, col_1[0], col_1[1], col_1[2]);
        t.setColor(2, col_2[0], col_2[1], col_2[2]);

        rasterize_triangle(t);

    }

    rst::rasterizer::MSAA();
    // MSAA();
}

void rst::rasterizer::MSAA()
{
    for(auto index : msaa_frame)
    {
        int midx = index.first / 701;
        int midy = index.first % 701;
        auto color = Eigen::Vector3f(0,0,0);
        for(int i = 0; i < 9 ; i ++)
        {
            if(msaa_depth[index.first][i] > depth_buf[get_index(midx , midy)] )
            {
                color += index.second[i];
            }
            else
            {
                color += frame_buf[get_index(midx,midy)];
            }
        }
        color /= 9;
        set_pixel(Eigen::Vector3f(midx,midy,1.0) , color);
    }
}


//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    
    //取得三个顶点坐标
    auto v = t.toVector4();

    int minx = std::min(v[0].x(),v[1].x());
    minx = std::min(minx , (int)v[2].x());

    int maxx = std::max(v[0].x(),v[1].x());
    maxx = std::max(maxx , (int)v[2].x());

    int miny = std::min(v[0].y(),v[1].y());
    miny = std::min(miny , (int)v[2].y());

    int maxy = std::max(v[0].y(),v[1].y());
    maxy = std::max(maxy , (int)v[2].y());

    
    //求三角形三边向量
    Vector3f side01 = t.v[1] - t.v[0];
    Vector3f side12 = t.v[2] - t.v[1];
    Vector3f side20 = t.v[0] - t.v[2];
    side01[2] = 0;
    side12[2] = 0;
    side20[2] = 0;

    for(int i = minx ; i <= maxx ; i++)
    {
        int midL = miny;
        int midR = maxy;
        while(midL <= midR && insideTriangle(i,midL,t.v,side01,side12,side20) == 0)
            midL ++;
        while(midL <= midR && insideTriangle(i,midR,t.v,side01,side12,side20) == 0)
            midR --;

        int msaaRange = 5;
        for(int j = midL - msaaRange ; j <= midR + msaaRange ; j++)
        {
            if(abs(j - midL) <= msaaRange || abs(j - midR) <= msaaRange)
            {
                if(msaa_depth.count(i*701+j) == 0)
                {
                    msaa_depth[(i*701 + j)] = std::vector<float>(9,std::numeric_limits<float>::infinity() * -1) ;
                    msaa_frame[(i*701 + j)] = std::vector< Eigen::Vector3f > (9 , Eigen::Vector3f{0,0,0} );
                }

                for(int offsetk = 0; offsetk < 9 ; offsetk ++)
                {
                    float midx = offsetX[offsetk] + i;
                    float midy = offsetY[offsetk] + j;
        
                    if(insideTriangle(midx,midy,t.v,side01,side12,side20) == 0)
                        continue; 
        
                    auto[alpha, beta, gamma] = computeBarycentric2D( midx , midy, t.v);
                    float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;

                    if(z_interpolated > msaa_depth[(i*701 + j)][offsetk])
                    {
                        msaa_depth[(i*701 + j)][offsetk] = z_interpolated;
                        msaa_frame[(i*701 + j)][offsetk] = t.getColor();
                    }

                }
            }
            else
            {
                
                auto[alpha, beta, gamma] = computeBarycentric2D(i, j, t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;


                if(z_interpolated > depth_buf[get_index(i,j)])
                {
                    depth_buf[get_index(i,j)] = z_interpolated;
                    set_pixel(Vector3f(i,j,1.0) , t.getColor());
                }
            }
        }
    }
    
    
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle

    // If so, use the following code to get the interpolated z value.
    //auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    //float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    //float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    //z_interpolated *= w_reciprocal;

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity() * -1);
    }

        msaa_depth.clear();
        msaa_frame.clear();
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h+1);
    depth_buf.resize(w * h+1);
    // msaa_depth.clear();
    // msaa_frame.clear();
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on
/*
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Author: Luca Risolia <info@linux-projects.org>
*/

#include <boost/exception/diagnostic_information.hpp>
#include <boost/program_options.hpp>
#include <boost/throw_exception.hpp>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

namespace po = boost::program_options;
namespace fs = std::filesystem;

namespace {
struct program_options_t {
  cv::Size board_size;
  unsigned int num_imgs;
  double square_size;
  bool top_bottom, show_rectified, show_corners, save_calib;
  fs::path imgs_dir, output_dir;
  std::string ext;
};

std::optional<program_options_t> parse_program_options(int argc, char **argv) {
  program_options_t opts;

  po::options_description desc{"Program options"};
  desc.add_options()("board-rows,r",
                     po::value<int>(&opts.board_size.width)->required(),
                     "number of rows in the board")(
      "board-cols,c", po::value<int>(&opts.board_size.height)->required(),
      "number of columns in the board")(
      "square-size,s", po::value<double>(&opts.square_size)->default_value(1.0),
      "number of rows in the board")(
      "num-imgs,n",
      po::value<unsigned int>(&opts.num_imgs)
          ->default_value(15)
          ->notifier([&opts](const auto &v) {
            if (v < 3) {
              boost::throw_exception(
                  std::runtime_error{"minimum for 'num_imgs' option is 3"});
            }
            opts.num_imgs = v;
          }),
      "maximum number of images with a properly identified pattern to "
      "consider")("imgs-dir,i",
                  po::value<std::string>()->required()->notifier(
                      [&opts](const auto &v) { opts.imgs_dir = v; }),
                  "path to the directory with the input stereo images")(
      "top-bottom,t",
      po::value<bool>(&opts.top_bottom)
          ->implicit_value(true)
          ->default_value(true),
      "left and right views in the images ara packed in a top/bottom layout")(
      "save-calibration,b",
      po::value<bool>(&opts.save_calib)
          ->implicit_value(true)
          ->default_value(true),
      "save calibration results in the output dir")(
      "output_dir,o",
      po::value<std::string>()
          ->default_value(fs::current_path().string())
          ->notifier([&opts](const auto &v) { opts.output_dir = v; }),
      "path to the output directory wherein the camera calibration results "
      "and/or the rectified input images are "
      "saved (when pressing 's' in the output window). Default path is current "
      "dir.")("show-rectified,u",
              po::value<bool>(&opts.show_rectified)
                  ->implicit_value(true)
                  ->default_value(true),
              "show undistorted and rectified images")(
      "show-corners,w",
      po::value<bool>(&opts.show_corners)
          ->implicit_value(true)
          ->default_value(true),
      "show detected corners")(
      "file-ext,e", po::value<std::string>(&opts.ext)->default_value(".png"),
      "parse images with given file exention only");

  po::options_description other_desc{"Other options"};
  other_desc.add_options()("help,h", "print help screen and exit");
  desc.add(other_desc);
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << '\n'
              << "example:\n stereo_calib -r 9 -c 6 -s 0.023 -n 15 -i "
                 "/path/to/imgs/ -e .png --top-bottom -o ./\n";
    return {};
  }

  return opts;
}

/*-----------------------------------------------------------------------------*/

struct view_t {
  cv::Mat img, grey_img;
  std::vector<cv::Point3f> obj_points;
  std::vector<cv::Point2f> img_points;
  std::string name;
};

using stereo_view_t = std::pair<view_t, view_t>;

struct calib_t {
  cv::Mat A1 = cv::Mat::eye(3, 3, CV_64F),
          A2 = cv::Mat::eye(
              3, 3, CV_64F), // to fix fx,fy ratio with CALIB_FIX_ASPECT_RATIO
      D1, D2, R, T, E, F;
  double reproject_err = 0;
};

std::vector<stereo_view_t>
parse_imgs(const fs::path &path, unsigned int num_imgs, cv::Size board_size,
           double square_size, bool top_bottom, fs::path output_dir,
           std::string_view ext = ".png", bool show = false) {
  if (!fs::is_directory(path)) {
    boost::throw_exception(
        std::runtime_error{path.string() + " is not directory"});
  }

  std::vector<stereo_view_t> stereo_imgs;

  fs::recursive_directory_iterator it{
      path, fs::directory_options::follow_directory_symlink};
  std::vector<fs::path> files{it, {}};
  std::sort(std::begin(files), std::end(files));
  for (unsigned n = 0; const fs::path &file : files) {
    if (file.extension() == ext) {
      std::cout << "loading stereo image '" << file << "'...";
      const auto stereo_img = cv::imread(file);
      const auto sz = stereo_img.size();
      std::cout << "image size is " << sz << '\n';
      view_t left, right;
      if (top_bottom) {
        left.img = stereo_img(cv::Rect{0, 0, sz.width, sz.height / 2});
        right.img =
            stereo_img(cv::Rect{0, sz.height / 2, sz.width, sz.height / 2});
      } else {
        left.img = stereo_img(cv::Rect{0, 0, sz.width / 2, sz.height});
        right.img =
            stereo_img(cv::Rect{0, sz.width / 2, sz.width / 2, sz.height});
      }
      if (bool found =
              cv::findChessboardCorners(left.img, board_size, left.img_points,
                                        cv::CALIB_CB_ADAPTIVE_THRESH |
                                            cv::CALIB_CB_FILTER_QUADS) &&
              cv::findChessboardCorners(right.img, board_size, right.img_points,
                                        cv::CALIB_CB_ADAPTIVE_THRESH |
                                            cv::CALIB_CB_FILTER_QUADS)) {
        n++;
        left.name = "left_" + file.stem().string();
        right.name = "right_" + file.stem().string();
        std::cout << "ok, patterns found!\n";
        for (auto view : {&left, &right}) {
          // Improve the accuracy with the grey-level image
          cv::cvtColor(view->img, view->grey_img, cv::COLOR_BGR2GRAY);
          cv::cornerSubPix(
              view->grey_img, view->img_points, cv::Size(5, 5),
              cv::Size(-1, -1),
              cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT,
                               60, 0.01));
          for (int i = 0; i < board_size.height; i++)
            for (int j = 0; j < board_size.width; j++)
              view->obj_points.emplace_back(j * square_size, i * square_size,
                                            0);
        }

        for (const auto &view : {left, right}) {
          if (show) {
            cv::Mat cimg;
            cv::cvtColor(view.grey_img, cimg, cv::COLOR_GRAY2BGR);
            drawChessboardCorners(cimg, board_size, view.img_points, found);
            cv::imshow("corners", cimg);
            if (const auto key = cv::waitKey();
                (key & 255) == 27 || key == 'q') {
              show = false;
            } else if (key == 's') {
              const fs::path save_as =
                  output_dir / (view.name + "_corners.png");
              std::cout << "saving image as " << save_as << "\n";
              if (!imwrite(save_as.string(), cimg))
                std::cerr << "error while saving\n";
            }
            cv::destroyWindow("corners");
          }
        }
        stereo_imgs.emplace_back(std::move(left), std::move(right));
      } else {
        std::cout << "expected patterns not found, skip...\n";
      }
      if (n >= num_imgs)
        break;
    }
  }
  return stereo_imgs;
}

calib_t stereo_calibrate(const std::vector<stereo_view_t> &stereo_imgs) {
  std::cout << "running stereo calibration...\n";
  BOOST_ASSERT(!stereo_imgs.empty());
  std::vector<decltype(view_t::obj_points)> all_obj_points;
  std::vector<decltype(view_t::img_points)> all_img_points_left,
      all_img_points_right;
  for (const auto &[left, right] : stereo_imgs) {
    BOOST_ASSERT(left.obj_points == right.obj_points);
    all_obj_points.push_back(left.obj_points);
    all_img_points_left.push_back(left.img_points);
    all_img_points_right.push_back(right.img_points);
  }
  calib_t calib;
  calib.reproject_err = cv::stereoCalibrate(
      all_obj_points, all_img_points_left, all_img_points_right, calib.A1,
      calib.D1, calib.A2, calib.D2, stereo_imgs.at(0).first.img.size(), calib.R,
      calib.T, calib.E, calib.F,
      // cv::CALIB_SAME_FOCAL_LENGTH |
      // cv::CALIB_FIX_PRINCIPAL_POINT |
      // cv::CALIB_ZERO_TANGENT_DIST |
      // cv::CALIB_RATIONAL_MODEL |
      cv::CALIB_FIX_ASPECT_RATIO,
      cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100,
                       1e-6));
  std::cout << "re-projection error: " << calib.reproject_err << "\n\n";
  std::cout << "First Camera Intrinsics:\n" << calib.A1 << "\n\n";
  std::cout << "First Camera Lens Distortion coefficients:\n"
            << calib.D1 << "\n\n";
  std::cout << "Second Camera Intrinsics:\n" << calib.A2 << "\n\n";
  std::cout << "Second Camera Lens Distortion coefficients:\n"
            << calib.D2 << "\n\n";
  std::cout << "Rotation Matrix:\n" << calib.R << "\n\n";
  std::cout << "Translation Matrix:\n" << calib.T << '\n';
  return calib;
}

std::vector<cv::Mat>
undistort_rectify(const std::vector<stereo_view_t> &stereo_imgs,
                  const calib_t &calib, fs::path output_dir, bool save,
                  bool show = true, bool use_sbgm = false) {
  BOOST_ASSERT(!stereo_imgs.empty());
  const auto img_size = stereo_imgs.at(0).first.img.size();
  cv::Mat R1, R2, P1, P2, map11, map12, map21, map22;
  cv::stereoRectify(calib.A1, calib.D1, calib.A2, calib.D2, img_size, calib.R,
                    calib.T, R1, R2, P1, P2, cv::noArray(),
                    cv::CALIB_ZERO_DISPARITY);
  cv::initUndistortRectifyMap(calib.A1, calib.D1, R1, P1, img_size, CV_16SC2,
                              map11, map12);
  cv::initUndistortRectifyMap(calib.A2, calib.D2, R2, P2, img_size, CV_16SC2,
                              map21, map22);
  if (save) {
    const auto file = output_dir / "stereo_calibration.xml";
    cv::FileStorage fs(file, cv::FileStorage::WRITE);
    fs << "camera_matrix_A1" << calib.A1 << "distortion_coefficients_D1"
       << calib.D1 << "camera_matrix_A2" << calib.A2
       << "distortion_coefficients_D2" << calib.D2 << "stereo_rotation_R"
       << calib.R << "stereo_translation_T" << calib.T
       << "rectified_rotation_R1" << R1 << "rectified_rotation_R2" << R2
       << "new_camera_matrix_1" << P1 << "new_camera_matrix_2" << P2;
  }
  std::vector<cv::Mat> rectified;
  cv::Mat pair;
  pair.create(img_size.height, img_size.width * 2, CV_8UC3);
  for (const auto &[left, right] : stereo_imgs) {
    cv::Mat img1r, img2r;
    cv::remap(left.grey_img, img1r, map11, map12, cv::INTER_LINEAR);
    cv::remap(right.grey_img, img2r, map21, map22, cv::INTER_LINEAR);
    cv::Mat part = pair.colRange(0, img_size.width);
    cv::cvtColor(img1r, part, cv::COLOR_GRAY2BGR);
    part = pair.colRange(img_size.width, img_size.width * 2);
    cv::cvtColor(img2r, part, cv::COLOR_GRAY2BGR);
    for (auto j = 0; j < img_size.height; j += 16) {
      cv::line(pair, cv::Point(0, j), cv::Point(img_size.width * 2, j),
               cv::Scalar(0, 255, 0));
    }
    if (show) {
      cv::imshow("rectified", pair);
      if (const auto key = cv::waitKey(); (key & 255) == 27 || key == 'q') {
        show = false;
      } else if (key == 's') {
        const fs::path left_save_as =
                           output_dir / (left.name + "_rectified.png"),
                       right_save_as =
                           output_dir / (right.name + "_rectified.png");
        std::cout << "saving image as " << left_save_as << " and "
                  << right_save_as << '\n';
        if (!imwrite(left_save_as.string(), img1r) ||
            !imwrite(right_save_as.string(), img2r))
          std::cerr << "error while saving\n";
      }
      cv::destroyWindow("rectified");
      if (show) {
        // Compute and show the disparity map
        cv::Ptr<cv::StereoMatcher> sm;
        if (use_sbgm) {
          // cv::Ptr<cv::StereoSGBM> sbgm = cv::StereoSGBM::create(); // defs.
          int block_size = 5,
              p1 = 600,  // doc suggests: 8 * 1 * block_size * block_size,
              p2 = 2400, // doc. suggets 32 * 1 * block_size * block_size,
              mindisp = -64, numdisp = 192, disp12maxdiff = 0, prefiltercap = 4,
              uniquenessratio = 1, specklewsize = 150, specklerange = 2;
          sm = cv::StereoSGBM::create(mindisp, numdisp, block_size, p1, p2,
                                      disp12maxdiff, prefiltercap,
                                      uniquenessratio, specklewsize,
                                      specklerange, cv::StereoSGBM::MODE_HH);
        } else {
          auto sbm = cv::StereoBM::create(0, 23);
          // sbm->setBlockSize(21);
          // sbm->setNumDisparities(112);
          // sbm->setPreFilterSize(7); // 7 should be the default
          // sbm->setPreFilterCap(63); // memb.func not virtualized
          // sbm->setTextureThreshold(500);
          // sbm->setMinDisparity(0);
          // sbm->setUniquenessRatio(15);
          // sbm->setSpeckleWindowSize(50);
          // sbm->setSpeckleRange(1);
          // sbm->setDisp12MaxDiff(1);
          sm = sbm;
        }
        cv::Mat disp;
        sm->compute(img1r, img2r, disp);
        cv::Mat ndisp;
        cv::normalize(disp, ndisp, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(ndisp, disp, cv::COLORMAP_JET);
        cv::imshow("disparity", disp);
        if (const auto key = cv::waitKey(); (key & 255) == 27 || key == 'q') {
          show = false;
        } else if (key == 's') {
          const fs::path save_as = output_dir / (left.name + "_disparity.png");
          std::cout << "saving image as " << save_as << "\n";
          if (!imwrite(save_as.string(), disp))
            std::cerr << "error while saving\n";
        }
        cv::destroyWindow("disparity");
      }
    }
    rectified.push_back(pair);
  }
  return rectified;
}

} // namespace

int main(int argc, char **argv) try {
  const auto opts = parse_program_options(argc, argv);

  if (!opts) {
    return EXIT_SUCCESS;
  }

  if (opts->show_corners || opts->show_rectified)
    std::cout
        << "To save the image shown in the window press 's', or 'q' to skip\n";

  const auto src_imgs = parse_imgs(
      opts->imgs_dir, opts->num_imgs, opts->board_size, opts->square_size,
      opts->top_bottom, opts->output_dir, opts->ext, opts->show_corners);

  const auto calib = stereo_calibrate(src_imgs);

  const auto rectified_imgs =
      undistort_rectify(src_imgs, calib, opts->output_dir, opts->save_calib,
                        opts->show_rectified);

  return EXIT_SUCCESS;

} catch (const po::error &err) {
  std::cerr << err.what() << '\n';
  return EXIT_FAILURE;

} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return EXIT_FAILURE;

} catch (...) {
  std::cerr << "unknown exception\n";
  return EXIT_FAILURE;
}

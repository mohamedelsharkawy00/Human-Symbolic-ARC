#pragma once
#define VK_USE_PLATFORM_WIN32_KHR
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include <windows.h>

#include "screenshot.hpp"

namespace arc {
using namespace std;

const int MAX_FRAMES_IN_FLIGHT = 1;

const vector<const char *> device_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_EXT_FULL_SCREEN_EXCLUSIVE_EXTENSION_NAME};

class vulkan_context {
public:
  vulkan_context();

  inline void init(HWND window, int width, int height);
  inline void recreate_swap_chain(int width, int height);

  inline void draw(int offset, int width, int height, int x = 0, int y = 0,
                   int cell_width = 0, int stride = 0);
  inline bool begin_frame();
  inline void end_frame();

  // A FRAME DRAWN INTO MEMORY, NOT ONTO THE WINDOW. The app runs minimized when
  // an agent drives it, and a minimized window renders nothing at all — so a
  // picture taken off the screen would be a picture of nothing, exactly when it
  // is most wanted. This path never touches the swapchain, so what it saves
  // does not depend on whether anyone can see the window.
  inline bool begin_capture(int width, int height);
  // An empty path throws the frame away — see why in the screenshot command.
  inline bool end_capture(const string &path);

  // Upload grid data to GPU SSBO
  inline void upload_all_data(const vector<int> &data, int actual_size = -1);

private:
  static const uint32_t STORAGE_BUFFER_SIZE = 256 * 1024 * 1024;

  struct push_constants {
    int grid_w;
    int grid_h;
    int screen_w;
    int screen_h;
    int data_offset;
    int cell_w;
    int start_x;
    int start_y;
    int stride;
  };

  inline void setup_device_context(HWND window);
  inline void aim_at_target(VkCommandBuffer buf, VkExtent2D size);
  inline void make_capture_target(int width, int height);
  inline void drop_capture_target();
  inline void setup_swapchain_resources(int width, int height);
  inline void create_pipeline();
  inline void create_execution_resources();
  inline void create_storage_buffer();
  inline void setup_descriptors();

  inline VkShaderModule create_shader_module(const vector<char> &code);
  inline void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags properties, VkBuffer &buffer,
                            VkDeviceMemory &buffer_memory);
  inline uint32_t find_memory_type(uint32_t filter,
                                   VkMemoryPropertyFlags props);

  VkInstance instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice physical_device = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue graphics_queue = VK_NULL_HANDLE;
  VkQueue present_queue = VK_NULL_HANDLE;
  VkSwapchainKHR swap_chain = VK_NULL_HANDLE;
  vector<VkImage> swap_chain_images;
  VkFormat swap_chain_image_format;
  VkExtent2D swap_chain_extent;
  vector<VkImageView> swap_chain_image_views;
  VkRenderPass render_pass = VK_NULL_HANDLE;

  VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline graphics_pipeline = VK_NULL_HANDLE;

  vector<VkFramebuffer> swap_chain_framebuffers;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  vector<VkCommandBuffer> command_buffers;
  vector<VkSemaphore> image_available_semaphores;
  vector<VkSemaphore> render_finished_semaphores;
  vector<VkFence> in_flight_fences;

  VkBuffer storage_buffer = VK_NULL_HANDLE;
  VkDeviceMemory storage_buffer_memory = VK_NULL_HANDLE;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

  uint32_t current_image_index = 0;
  uint32_t current_frame = 0;

  // WHAT IS BEING DRAWN INTO RIGHT NOW. draw() used to name the swapchain
  // itself, which left no way to aim the same drawing anywhere else. Both
  // paths set these two and draw() reads them, so there is one drawing routine
  // and not one per destination.
  VkCommandBuffer active_command = VK_NULL_HANDLE;
  VkExtent2D active_extent{};

  // Where a captured frame is drawn and then read back from. Made on the first
  // capture and remade only when the size asked for changes.
  VkExtent2D capture_extent{};
  VkImage capture_image = VK_NULL_HANDLE;
  VkDeviceMemory capture_image_memory = VK_NULL_HANDLE;
  VkImageView capture_view = VK_NULL_HANDLE;
  VkFramebuffer capture_framebuffer = VK_NULL_HANDLE;
  VkBuffer capture_buffer = VK_NULL_HANDLE;
  VkDeviceMemory capture_buffer_memory = VK_NULL_HANDLE;
  VkCommandBuffer capture_command = VK_NULL_HANDLE;
  VkFence capture_fence = VK_NULL_HANDLE;

  HWND hwnd = nullptr;
  // Set when acquire/present reports the swapchain is out of date or
  // suboptimal. The owner recreates on the next frame instead of polling a
  // GUI flag (which previously fed a 0x0 / minimized extent into the driver
  // and AV'd inside nvwgf2umx on alt-tab).
  bool framebuffer_resized = false;

public:
  // True when the window is minimized or has a zero-size client area. The
  // caller must skip rendering AND swapchain recreation in that state: a
  // zero imageExtent is invalid per the Vulkan spec and crashes the NV
  // driver. DWM keeps the last presented frame for the taskbar thumbnail,
  // so pausing does not lose the hover preview.
  inline bool minimized() const {
    if (!hwnd || IsIconic(hwnd)) return true;
    RECT rc{};
    if (!GetClientRect(hwnd, &rc)) return true;
    return (rc.right - rc.left) <= 0 || (rc.bottom - rc.top) <= 0;
  }
  inline bool needs_recreate() const { return framebuffer_resized; }
  inline void client_extent(int &w, int &h) const {
    RECT rc{};
    if (hwnd && GetClientRect(hwnd, &rc)) {
      w = rc.right - rc.left;
      h = rc.bottom - rc.top;
    } else {
      w = 0;
      h = 0;
    }
  }
};

// ==========================================
// IMPLEMENTATION
// ==========================================

inline vulkan_context::vulkan_context()
    : swap_chain_image_format(VK_FORMAT_B8G8R8A8_UNORM),
      swap_chain_extent({0, 0}) {}

inline void vulkan_context::recreate_swap_chain(int width, int height) {
  // A zero extent (minimized window) is an invalid imageExtent and crashes
  // the NV driver. Never recreate in that state; the caller's minimized()
  // guard should already prevent reaching here, this is defense in depth.
  if (width <= 0 || height <= 0)
    return;
  framebuffer_resized = false;
  vkDeviceWaitIdle(device);
  for (auto fb : swap_chain_framebuffers)
    vkDestroyFramebuffer(device, fb, 0);
  if (graphics_pipeline != VK_NULL_HANDLE) {
    vkDestroyPipeline(device, graphics_pipeline, 0);
    vkDestroyPipelineLayout(device, pipeline_layout, 0);
  }
  vkDestroyRenderPass(device, render_pass, 0);
  for (auto iv : swap_chain_image_views)
    vkDestroyImageView(device, iv, 0);
  vkDestroySwapchainKHR(device, swap_chain, 0);
  drop_capture_target();
  setup_swapchain_resources(width, height);
  create_pipeline();
}

inline void vulkan_context::init(HWND window, int width, int height) {
  setup_device_context(window);
  create_storage_buffer();
  setup_descriptors();
  setup_swapchain_resources(width, height);
  create_pipeline();
  create_execution_resources();
}

inline void vulkan_context::setup_device_context(HWND window) {
  hwnd = window;
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO,
                        nullptr,
                        "ARC",
                        1,
                        "No Engine",
                        1,
                        VK_API_VERSION_1_0};
  vector<const char *> exts = {"VK_KHR_surface", "VK_KHR_win32_surface"};
  VkInstanceCreateInfo inst_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                 nullptr,
                                 0,
                                 &app,
                                 0,
                                 nullptr,
                                 (uint32_t)exts.size(),
                                 exts.data()};
  vkCreateInstance(&inst_info, nullptr, &instance);

  VkWin32SurfaceCreateInfoKHR surf_info{
      VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR, nullptr, 0,
      GetModuleHandle(nullptr), window};
  vkCreateWin32SurfaceKHR(instance, &surf_info, nullptr, &surface);

  uint32_t count = 0;
  vkEnumeratePhysicalDevices(instance, &count, nullptr);
  vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(instance, &count, devices.data());
  physical_device = devices[0];

  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
  vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count,
                                           families.data());
  uint32_t q_idx = 0;
  for (uint32_t i = 0; i < count; i++) {
    VkBool32 push;
    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, i, surface, &push);
    if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && push) {
      q_idx = i;
      break;
    }
  }

  float q_prio = 1.0f;
  VkDeviceQueueCreateInfo q_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                 nullptr,
                                 0,
                                 q_idx,
                                 1,
                                 &q_prio};
  VkDeviceCreateInfo d_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                            nullptr,
                            0,
                            1,
                            &q_info,
                            0,
                            nullptr,
                            (uint32_t)device_extensions.size(),
                            device_extensions.data()};
  vkCreateDevice(physical_device, &d_info, nullptr, &device);
  vkGetDeviceQueue(device, q_idx, 0, &graphics_queue);
  vkGetDeviceQueue(device, q_idx, 0, &present_queue);
}

inline void vulkan_context::setup_descriptors() {
  VkDescriptorSetLayoutBinding bind{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                    VK_SHADER_STAGE_VERTEX_BIT};
  VkDescriptorSetLayoutCreateInfo l_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 1, &bind};
  vkCreateDescriptorSetLayout(device, &l_info, 0, &descriptor_set_layout);

  VkDescriptorPoolSize p_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
  VkDescriptorPoolCreateInfo p_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, 0, 0, 1, 1, &p_size};
  vkCreateDescriptorPool(device, &p_info, 0, &descriptor_pool);

  VkDescriptorSetAllocateInfo a_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0, descriptor_pool, 1,
      &descriptor_set_layout};
  vkAllocateDescriptorSets(device, &a_info, &descriptor_set);

  VkDescriptorBufferInfo b_info{storage_buffer, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet write{
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, descriptor_set, 0, 0, 1,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,      0, &b_info};
  vkUpdateDescriptorSets(device, 1, &write, 0, 0);
}

inline void vulkan_context::setup_swapchain_resources(int width, int height) {
  VkSurfaceCapabilitiesKHR caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &caps);
  swap_chain_extent = {clamp((uint32_t)width, caps.minImageExtent.width,
                             caps.maxImageExtent.width),
                       clamp((uint32_t)height, caps.minImageExtent.height,
                             caps.maxImageExtent.height)};

  uint32_t image_count =
      clamp(caps.minImageCount + 1, caps.minImageCount,
            caps.maxImageCount > 0 ? caps.maxImageCount : 0xFFFFFFFF);
  VkSwapchainCreateInfoKHR sc_info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  sc_info.surface = surface;
  sc_info.minImageCount = image_count;
  sc_info.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
  sc_info.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  sc_info.imageExtent = swap_chain_extent;
  sc_info.imageArrayLayers = 1;
  sc_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  sc_info.preTransform = caps.currentTransform;
  sc_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  sc_info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  sc_info.clipped = VK_TRUE;

  VkSurfaceFullScreenExclusiveInfoEXT fse{
      VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT, nullptr,
      VK_FULL_SCREEN_EXCLUSIVE_ALLOWED_EXT};
  sc_info.pNext = &fse;
  vkCreateSwapchainKHR(device, &sc_info, nullptr, &swap_chain);

  vkGetSwapchainImagesKHR(device, swap_chain, &image_count, nullptr);
  swap_chain_images.resize(image_count);
  vkGetSwapchainImagesKHR(device, swap_chain, &image_count,
                          swap_chain_images.data());

  swap_chain_image_views.resize(image_count);
  for (uint32_t i = 0; i < image_count; i++) {
    VkImageViewCreateInfo iv_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                  nullptr,
                                  0,
                                  swap_chain_images[i],
                                  VK_IMAGE_VIEW_TYPE_2D,
                                  VK_FORMAT_B8G8R8A8_UNORM};
    iv_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(device, &iv_info, nullptr, &swap_chain_image_views[i]);
  }

  VkAttachmentDescription color_att{};
  color_att.format = VK_FORMAT_B8G8R8A8_UNORM;
  color_att.samples = VK_SAMPLE_COUNT_1_BIT;
  color_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color_att.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_ref;
  VkSubpassDependency dep{};
  dep.srcSubpass = VK_SUBPASS_EXTERNAL;
  dep.dstSubpass = 0;
  dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  VkRenderPassCreateInfo rp_info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                                 nullptr,
                                 0,
                                 1,
                                 &color_att,
                                 1,
                                 &subpass,
                                 1,
                                 &dep};
  vkCreateRenderPass(device, &rp_info, nullptr, &render_pass);

  swap_chain_framebuffers.resize(image_count);
  for (uint32_t i = 0; i < image_count; i++) {
    VkFramebufferCreateInfo fb_info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                                    nullptr,
                                    0,
                                    render_pass,
                                    1,
                                    &swap_chain_image_views[i],
                                    swap_chain_extent.width,
                                    swap_chain_extent.height,
                                    1};
    vkCreateFramebuffer(device, &fb_info, nullptr, &swap_chain_framebuffers[i]);
  }
}

inline VkShaderModule
vulkan_context::create_shader_module(const vector<char> &code) {
  VkShaderModuleCreateInfo c{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0,
                             code.size(), (uint32_t *)code.data()};
  VkShaderModule m;
  vkCreateShaderModule(device, &c, 0, &m);
  return m;
}

static vector<char> read_file(const string &fn) {
  ifstream f(fn, ios::ate | ios::binary);
  if (!f.is_open()) {
    // Fallback: if "build/file.spv" fails, try "file.spv" (CMake artifacts
    // often colocated with exe)
    if (fn.substr(0, 6) == "build/") {
      f.open(fn.substr(6), ios::ate | ios::binary);
    }
  }
  if (!f.is_open()) {
    throw runtime_error("Failed to open file: " + fn);
  }
  auto tell = f.tellg();
  if (tell < 0) {
    throw runtime_error("Failed to determine file size: " + fn);
  }
  vector<char> b((size_t)tell);
  f.seekg(0);
  f.read(b.data(), b.size());
  return b;
}
inline void vulkan_context::create_pipeline() {
  VkShaderModule v_mod = create_shader_module(read_file("out/vert.spv")),
                 f_mod = create_shader_module(read_file("out/frag.spv"));
  VkPipelineShaderStageCreateInfo stages[] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0,
       VK_SHADER_STAGE_VERTEX_BIT, v_mod, "main"},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0,
       VK_SHADER_STAGE_FRAGMENT_BIT, f_mod, "main"}};
  VkPipelineVertexInputStateCreateInfo v_in{
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ia{
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, 0, 0,
      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
  // WHERE TO DRAW IS SAID PER FRAME, NOT BAKED IN. Baked, it was the size the
  // window happened to be when the pipeline was built — and a window that
  // starts minimized has no size at all, so every draw fell outside the
  // viewport and the app rendered nothing anyone could see. Said per frame it
  // is whatever is being drawn into, window or memory.
  VkPipelineViewportStateCreateInfo vs{
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      0, 0, 1, nullptr, 1, nullptr};
  const VkDynamicState said_per_frame[] = { VK_DYNAMIC_STATE_VIEWPORT,
                                            VK_DYNAMIC_STATE_SCISSOR };
  VkPipelineDynamicStateCreateInfo dyn{
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      0, 0, 2, said_per_frame};
  VkPipelineRasterizationStateCreateInfo rs{
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.lineWidth = 1.0f;
  rs.cullMode = VK_CULL_MODE_NONE;
  rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
  VkPipelineMultisampleStateCreateInfo ms{
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, 0, 0,
      VK_SAMPLE_COUNT_1_BIT};
  VkPipelineColorBlendAttachmentState cba{};
  cba.colorWriteMask = 0xF;
  cba.blendEnable = VK_TRUE;
  cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  cba.colorBlendOp = VK_BLEND_OP_ADD;
  cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  cba.alphaBlendOp = VK_BLEND_OP_ADD;
  VkPipelineColorBlendStateCreateInfo cb{
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      0,
      0,
      0,
      (VkLogicOp)0,
      1,
      &cba};
  VkPushConstantRange pr{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push_constants)};
  VkPipelineLayoutCreateInfo l_i{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                 0,
                                 0,
                                 1,
                                 &descriptor_set_layout,
                                 1,
                                 &pr};
  vkCreatePipelineLayout(device, &l_i, 0, &pipeline_layout);
  VkGraphicsPipelineCreateInfo p_i{
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      0,
      0,
      2,
      stages,
      &v_in,
      &ia,
      0,
      &vs,
      &rs,
      &ms,
      0,
      &cb,
      &dyn,
      pipeline_layout,
      render_pass,
      0};
  vkCreateGraphicsPipelines(device, 0, 1, &p_i, 0, &graphics_pipeline);
  vkDestroyShaderModule(device, v_mod, 0);
  vkDestroyShaderModule(device, f_mod, 0);
}

inline uint32_t vulkan_context::find_memory_type(uint32_t filter,
                                                 VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mem;
  vkGetPhysicalDeviceMemoryProperties(physical_device, &mem);
  for (uint32_t i = 0; i < mem.memoryTypeCount; i++)
    if ((filter & (1 << i)) &&
        (mem.memoryTypes[i].propertyFlags & props) == props)
      return i;
  return 0;
}

inline void vulkan_context::create_buffer(VkDeviceSize size,
                                          VkBufferUsageFlags use,
                                          VkMemoryPropertyFlags props,
                                          VkBuffer &buf, VkDeviceMemory &mem) {
  VkBufferCreateInfo b_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            0,
                            0,
                            size,
                            use,
                            VK_SHARING_MODE_EXCLUSIVE};
  vkCreateBuffer(device, &b_info, 0, &buf);
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(device, buf, &req);
  VkMemoryAllocateInfo a_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0,
                              req.size,
                              find_memory_type(req.memoryTypeBits, props)};
  vkAllocateMemory(device, &a_info, 0, &mem);
  vkBindBufferMemory(device, buf, mem, 0);
}

inline void vulkan_context::create_storage_buffer() {
  create_buffer(STORAGE_BUFFER_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                storage_buffer, storage_buffer_memory);
}

inline void vulkan_context::upload_all_data(const vector<int> &data,
                                            int actual_size) {
  if (data.empty())
    return;
  size_t upload_size = (actual_size > 0) ? (size_t)actual_size * sizeof(int)
                                         : data.size() * sizeof(int);

  if (upload_size > STORAGE_BUFFER_SIZE) {
    throw runtime_error("Data exceeds storage buffer size!");
  }

  void *p_data;
  vkMapMemory(device, storage_buffer_memory, 0, upload_size, 0, &p_data);
  memcpy(p_data, data.data(), upload_size);
  vkUnmapMemory(device, storage_buffer_memory);
}

inline void vulkan_context::create_execution_resources() {
  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = 0; // standard setup

  if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) !=
      VK_SUCCESS)
    throw runtime_error("failed to create command pool!");

  command_buffers.resize(MAX_FRAMES_IN_FLIGHT);
  VkCommandBufferAllocateInfo alloc_info{
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc_info.commandPool = command_pool;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = (uint32_t)command_buffers.size();

  if (vkAllocateCommandBuffers(device, &alloc_info, command_buffers.data()) !=
      VK_SUCCESS)
    throw runtime_error("failed to allocate command buffers!");

  image_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
  render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
  in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);

  VkSemaphoreCreateInfo sem_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr,
                               VK_FENCE_CREATE_SIGNALED_BIT};

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    if (vkCreateSemaphore(device, &sem_info, nullptr,
                          &image_available_semaphores[i]) != VK_SUCCESS ||
        vkCreateSemaphore(device, &sem_info, nullptr,
                          &render_finished_semaphores[i]) != VK_SUCCESS ||
        vkCreateFence(device, &fence_info, nullptr, &in_flight_fences[i]) !=
            VK_SUCCESS) {
      throw runtime_error("failed to create sync objects!");
    }
  }
}

// The size being drawn into, said to the command buffer that will do it.
inline void vulkan_context::aim_at_target(VkCommandBuffer buf, VkExtent2D size) {
  VkViewport where{0, 0, (float)size.width, (float)size.height, 0, 1};
  VkRect2D all{{0, 0}, size};
  vkCmdSetViewport(buf, 0, 1, &where);
  vkCmdSetScissor(buf, 0, 1, &all);
}

inline bool vulkan_context::begin_frame() {
  vkWaitForFences(device, 1, &in_flight_fences[current_frame], 1, UINT64_MAX);
  VkResult acquire_result = vkAcquireNextImageKHR(device, swap_chain, UINT64_MAX,
                            image_available_semaphores[current_frame], 0,
                            &current_image_index);
  if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
    framebuffer_resized = true;
    return false;
  }
  if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR)
    return false;
  vkResetFences(device, 1, &in_flight_fences[current_frame]);

  VkCommandBuffer buf = command_buffers[current_frame];
  active_command = buf;
  active_extent = swap_chain_extent;
  vkResetCommandBuffer(buf, 0);
  VkCommandBufferBeginInfo b_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(buf, &b_info);

  VkClearValue clear{{{0, 0, 0, 1}}};
  VkRenderPassBeginInfo rp_info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                0,
                                render_pass,
                                swap_chain_framebuffers[current_image_index],
                                {{0, 0}, swap_chain_extent},
                                1,
                                &clear};
  vkCmdBeginRenderPass(buf, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(buf, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);
  vkCmdBindDescriptorSets(buf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                          0, 1, &descriptor_set, 0, 0);
  aim_at_target(buf, swap_chain_extent);
  return true;
}

inline void vulkan_context::draw(int offset, int width, int height, int x,
                                 int y, int cell_w, int stride) {
  if (active_command == VK_NULL_HANDLE) return;
  push_constants p{width,
                   height,
                   (int)active_extent.width,
                   (int)active_extent.height,
                   offset,
                   cell_w,
                   x,
                   y,
                   stride > 0 ? stride : width};
  vkCmdPushConstants(active_command, pipeline_layout,
                     VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(p), &p);
  vkCmdDraw(active_command, 6, width * height, 0, 0);
}

inline void vulkan_context::end_frame() {
  VkCommandBuffer buf = command_buffers[current_frame];
  vkCmdEndRenderPass(buf);
  vkEndCommandBuffer(buf);

  VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo s_info{VK_STRUCTURE_TYPE_SUBMIT_INFO,
                      0,
                      1,
                      &image_available_semaphores[current_frame],
                      &wait,
                      1,
                      &buf,
                      1,
                      &render_finished_semaphores[current_frame]};
  vkQueueSubmit(graphics_queue, 1, &s_info, in_flight_fences[current_frame]);
  active_command = VK_NULL_HANDLE;

  VkSwapchainKHR sc = swap_chain;
  VkPresentInfoKHR p_info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                          0,
                          1,
                          &render_finished_semaphores[current_frame],
                          1,
                          &sc,
                          &current_image_index};
  VkResult present_result = vkQueuePresentKHR(present_queue, &p_info);
  if (present_result == VK_ERROR_OUT_OF_DATE_KHR ||
      present_result == VK_SUBOPTIMAL_KHR)
    framebuffer_resized = true;
  current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// The image a captured frame is drawn into, the buffer it is read back through,
// and the one command buffer that does both. Sized to what was asked for.
inline void vulkan_context::make_capture_target(int width, int height) {
  drop_capture_target();
  capture_extent = { (uint32_t)width, (uint32_t)height };

  VkImageCreateInfo img{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  img.imageType = VK_IMAGE_TYPE_2D;
  img.format = VK_FORMAT_B8G8R8A8_UNORM;
  img.extent = { capture_extent.width, capture_extent.height, 1 };
  img.mipLevels = 1;
  img.arrayLayers = 1;
  img.samples = VK_SAMPLE_COUNT_1_BIT;
  img.tiling = VK_IMAGE_TILING_OPTIMAL;
  img.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  img.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  vkCreateImage(device, &img, nullptr, &capture_image);

  VkMemoryRequirements need;
  vkGetImageMemoryRequirements(device, capture_image, &need);
  VkMemoryAllocateInfo give{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, need.size,
      find_memory_type(need.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
  vkAllocateMemory(device, &give, nullptr, &capture_image_memory);
  vkBindImageMemory(device, capture_image, capture_image_memory, 0);

  VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0,
                             capture_image, VK_IMAGE_VIEW_TYPE_2D,
                             VK_FORMAT_B8G8R8A8_UNORM};
  view.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
  vkCreateImageView(device, &view, nullptr, &capture_view);

  VkFramebufferCreateInfo frame{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, nullptr, 0,
                                render_pass, 1, &capture_view,
                                capture_extent.width, capture_extent.height, 1};
  vkCreateFramebuffer(device, &frame, nullptr, &capture_framebuffer);

  create_buffer((VkDeviceSize)capture_extent.width * capture_extent.height * 4,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                capture_buffer, capture_buffer_memory);

  VkCommandBufferAllocateInfo one{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                  nullptr, command_pool,
                                  VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
  vkAllocateCommandBuffers(device, &one, &capture_command);
  VkFenceCreateInfo done{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  vkCreateFence(device, &done, nullptr, &capture_fence);
}

inline void vulkan_context::drop_capture_target() {
  if (capture_fence) vkDestroyFence(device, capture_fence, nullptr);
  if (capture_command) vkFreeCommandBuffers(device, command_pool, 1, &capture_command);
  if (capture_buffer) vkDestroyBuffer(device, capture_buffer, nullptr);
  if (capture_buffer_memory) vkFreeMemory(device, capture_buffer_memory, nullptr);
  if (capture_framebuffer) vkDestroyFramebuffer(device, capture_framebuffer, nullptr);
  if (capture_view) vkDestroyImageView(device, capture_view, nullptr);
  if (capture_image) vkDestroyImage(device, capture_image, nullptr);
  if (capture_image_memory) vkFreeMemory(device, capture_image_memory, nullptr);
  capture_fence = VK_NULL_HANDLE;
  capture_command = VK_NULL_HANDLE;
  capture_buffer = VK_NULL_HANDLE;
  capture_buffer_memory = VK_NULL_HANDLE;
  capture_framebuffer = VK_NULL_HANDLE;
  capture_view = VK_NULL_HANDLE;
  capture_image = VK_NULL_HANDLE;
  capture_image_memory = VK_NULL_HANDLE;
  capture_extent = {};
}

inline bool vulkan_context::begin_capture(int width, int height) {
  if (device == VK_NULL_HANDLE || width <= 0 || height <= 0) return false;
  if (capture_image == VK_NULL_HANDLE ||
      capture_extent.width != (uint32_t)width ||
      capture_extent.height != (uint32_t)height)
    make_capture_target(width, height);

  vkResetCommandBuffer(capture_command, 0);
  VkCommandBufferBeginInfo start{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(capture_command, &start);
  VkClearValue clear{{{0, 0, 0, 1}}};
  VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, nullptr,
                             render_pass, capture_framebuffer,
                             {{0, 0}, capture_extent}, 1, &clear};
  vkCmdBeginRenderPass(capture_command, &pass, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(capture_command, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);
  vkCmdBindDescriptorSets(capture_command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipeline_layout, 0, 1, &descriptor_set, 0, 0);
  aim_at_target(capture_command, capture_extent);
  active_command = capture_command;
  active_extent = capture_extent;
  return true;
}

inline bool vulkan_context::end_capture(const string &path) {
  if (active_command != capture_command) return false;
  vkCmdEndRenderPass(capture_command);
  // A frame drawn only to find out what it wanted. Nothing reads it, so it is
  // never copied back and never sent to the card.
  if (path.empty()) {
    vkEndCommandBuffer(capture_command);
    active_command = VK_NULL_HANDLE;
    return true;
  }

  // The render pass leaves it as a thing to show; a copy wants a thing to read.
  VkImageMemoryBarrier to_read{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  to_read.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  to_read.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  to_read.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  to_read.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  to_read.image = capture_image;
  to_read.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
  vkCmdPipelineBarrier(capture_command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                       1, &to_read);

  VkBufferImageCopy copy{};
  copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
  copy.imageExtent = { capture_extent.width, capture_extent.height, 1 };
  vkCmdCopyImageToBuffer(capture_command, capture_image,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, capture_buffer, 1, &copy);
  vkEndCommandBuffer(capture_command);
  active_command = VK_NULL_HANDLE;

  vkResetFences(device, 1, &capture_fence);
  VkSubmitInfo send{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  send.commandBufferCount = 1;
  send.pCommandBuffers = &capture_command;
  if (vkQueueSubmit(graphics_queue, 1, &send, capture_fence) != VK_SUCCESS) return false;
  vkWaitForFences(device, 1, &capture_fence, VK_TRUE, UINT64_MAX);

  void *pixels = nullptr;
  if (vkMapMemory(device, capture_buffer_memory, 0, VK_WHOLE_SIZE, 0, &pixels) != VK_SUCCESS)
    return false;
  const bool wrote = write_png(path, (int)capture_extent.width,
                               (int)capture_extent.height, (const uint8_t *)pixels,
                               (int)capture_extent.width * 4);
  vkUnmapMemory(device, capture_buffer_memory);
  return wrote;
}
}

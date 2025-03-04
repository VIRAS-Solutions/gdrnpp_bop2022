//g++ egl.cpp -lpthread -ldl -lGL
#include <stdio.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>

// NOTE: these are usually in /usr/include/EGL, /usr/include/GL
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define MAX_NUM_RESOURCES 10

namespace py = pybind11;

struct EGLInternalData2 {
    bool m_isInitialized;

    int m_windowWidth;
    int m_windowHeight;
    int m_renderDevice;

    EGLBoolean success;
    EGLint num_configs;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;

    EGLInternalData2()
    : m_isInitialized(false),
    m_windowWidth(0),
    m_windowHeight(0) {}
};


static EGLDisplay getCudaDisplay(int cudaDeviceIdx)
{
    typedef EGLBoolean (*eglQueryDevicesEXT_t)(EGLint, EGLDeviceEXT, EGLint*);
    typedef EGLBoolean (*eglQueryDeviceAttribEXT_t)(EGLDeviceEXT, EGLint, EGLAttrib*);
    typedef EGLDisplay (*eglGetPlatformDisplayEXT_t)(EGLenum, void*, const EGLint*);
    std::cout << "CudaDeviceIDX: " << cudaDeviceIdx << std::endl;

    eglQueryDevicesEXT_t eglQueryDevicesEXT = (eglQueryDevicesEXT_t)eglGetProcAddress("eglQueryDevicesEXT");
    if (!eglQueryDevicesEXT)
    {
        fprintf(stderr, "eglGetProcAddress(\"eglQueryDevicesEXT\") failed\n\n\n");
        std::cout << "ERRRRROOOOORRRRR" << std::endl ;
        return 0;
    }
    // Testaufruf der Funktion
    EGLint maxDevices1 = 10; // Maximale Anzahl von Geräten, die abgerufen werden sollen
    EGLDeviceEXT devices1[10]; // Array zum Speichern der Geräte
    EGLint numDevices1; // Variable zum Speichern der Anzahl von zurückgegebenen Geräten
    EGLBoolean result1 = eglQueryDevicesEXT(maxDevices1, devices1, &numDevices1);
    if (result1 == EGL_FALSE) {
        std::cout << "Fehler: eglQueryDevicesEXT konnte nicht erfolgreich ausgeführt werden.\n";
    }
    eglQueryDeviceAttribEXT_t eglQueryDeviceAttribEXT = (eglQueryDeviceAttribEXT_t)eglGetProcAddress("eglQueryDeviceAttribEXT");
    std::cout << "TESTESTESTESTESTESTESTJGHGHJ" << *eglQueryDeviceAttribEXT << eglQueryDeviceAttribEXT << std::endl;
    if (eglQueryDeviceAttribEXT == NULL)
    {
        fprintf(stderr, "eglGetProcAddress(\"eglQueryDeviceAttribEXT\") failed\n\n\n");
        std::cout << "ERRRRROOOOORRRRR" << std::endl ;
        return 0;
    }

    eglGetPlatformDisplayEXT_t eglGetPlatformDisplayEXT = (eglGetPlatformDisplayEXT_t)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!eglGetPlatformDisplayEXT)
    {
        fprintf(stderr, "eglGetProcAddress(\"eglGetPlatformDisplayEXT\") failed\n\n\n");
        return 0;
    }

    int num_devices = 0;
    EGLBoolean result = eglQueryDevicesEXT(0, 0, &num_devices);
    std::cout << result << std::endl;
    if (result == EGL_FALSE){
        std::cout << "no EGL devices found\n";
    }
    if (!num_devices)
    {
        fprintf(stderr,"No EGL Devices Extensions regsiertered");
        std::cout << "ERRRRROOOOORRRRR" << std::endl ;
        return 0;
    }
    EGLDisplay display = 0;
    EGLDeviceEXT* devices = (EGLDeviceEXT*)malloc(num_devices * sizeof(void*));
    std::cout << "EGL STATUS" << eglQueryDevicesEXT(num_devices, devices, &num_devices) << std::endl;
    std::cout << "->" << num_devices << "," << devices << "," << num_devices << std::endl;
    printf("num = %d \n", num_devices);
    for (int i=0; i < num_devices; i++)
    {

        EGLDeviceEXT device = devices[i];
        intptr_t value = -1;
        
        //auto status = eglQueryDeviceAttribEXT(device, EGL_CUDA_DEVICE_NV, &value);
        EGLBoolean status = eglQueryDeviceAttribEXT(device, EGL_CUDA_DEVICE_NV, &value);
        if(status == EGL_FALSE){
            std::cout << "Fehler: eglQueryDeviceAttribEXT konnte nicht erfolgreich ausgeführt werden.\n" << std::endl;
        }
        
        EGLint errorCode = eglGetError();
        if(!status){
            std::cout << "ERROR CODE: " << errorCode <<std::endl;
        }
        if(errorCode == EGL_BAD_ATTRIBUTE){
            std::cout << "BAD ATTRIBUTE\n";
        }
        if(errorCode == EGL_BAD_DISPLAY){
            std::cout << "EGL_BAD_DISPLAY\n";
        }
        if(errorCode == EGL_CONTEXT_LOST){
            std::cout << "EGL_CONTEXT_LOST\n";
        }
        if(errorCode == EGL_BAD_ACCESS){
            std::cout << "EGL_BAD_ACCESS\n";
        }
        if(errorCode == EGL_NOT_INITIALIZED){
            std::cout << "EGL_NOT_INITIALIZED\n";
        }
        if(errorCode == EGL_BAD_ALLOC){
            std::cout << "EGL_BAD_ALLOC\n";
        }
        std::cout << "STATUS: " << status << " , " << value << " , " << device << std::endl ;
        std::cout << eglQueryDeviceAttribEXT << std::endl;
        if (status )//&& value == cudaDeviceIdx)
        {
            display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, device, 0);
            std::cout << "reingegangenSSSSSSSSSSSSSSSSSSS" << std::endl;
            break;
        }
    }
    free(devices);
    std::cout << "DDDDDDDDDDIIIIIIIIIIIISSSSSSSSSSSPPPPPPPPPPAAAALLLLLYS:" << display <<std::endl;
    return display;
}


class CppEGLRenderer{
public:
    CppEGLRenderer(int w, int h, int d):m_windowHeight(h),m_windowWidth(w),m_renderDevice(d) {};

    int m_windowWidth;
    int m_windowHeight;
    int m_renderDevice;

    EGLBoolean success;
    EGLint num_configs;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;


    EGLInternalData2* m_data = NULL;

    cudaGraphicsResource* cuda_res[MAX_NUM_RESOURCES];


    int init() {

        m_data = new EGLInternalData2();

        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(m_renderDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            exit(EXIT_FAILURE);
        } 
        std::cout <<  "cudaSetDevice SET DEVICE SUCCESSFULLLL\n" << std::endl;


        EGLint egl_config_attribs[] = {
            EGL_RED_SIZE,           8,
            EGL_GREEN_SIZE,         8,
            EGL_BLUE_SIZE,          8,
            EGL_DEPTH_SIZE,         24, // 8,
            EGL_STENCIL_SIZE,       8,
            EGL_RENDERABLE_TYPE,    EGL_OPENGL_BIT,
            EGL_SURFACE_TYPE,       EGL_PBUFFER_BIT,
            EGL_COLOR_BUFFER_TYPE,  EGL_RGB_BUFFER,
            EGL_RENDERABLE_TYPE,    EGL_OPENGL_BIT,
            EGL_CONFORMANT,         EGL_OPENGL_BIT,
            EGL_NONE
        };

        for (int i = 0; i < MAX_NUM_RESOURCES; i++)
            cuda_res[i] = NULL;

        m_data->m_renderDevice = m_renderDevice;
        // Query EGL Screens
        if (m_renderDevice >= 0) {
            char pciBusId[256] = "";
            printf("Creating GL context for Cuda device %d\n;", m_renderDevice);
            m_data->egl_display = getCudaDisplay(m_renderDevice);
            fprintf(stderr,"TTTstststststs");
            std::cout << "m_dtata: " << m_data->egl_display << std::endl;
            if (!m_data->egl_display)
                fprintf(stderr, "Failed, falling back to default display");
        }



        if (!m_data->egl_display)
        {
            m_data->egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
            if (m_data->egl_display == EGL_NO_DISPLAY) {
                fprintf(stderr, "eglGetDisplay() failed");
                exit(EXIT_FAILURE);
            }
        }

        EGLint major, minor;
        if (!eglInitialize(m_data->egl_display, &major, &minor)) {
            fprintf(stderr, "Unable to initialize EGL\n");
            exit(EXIT_FAILURE);
        }

        printf("(egl_renderer) Loaded EGL version: %d.%d.\n", major, minor);

        m_data->success = eglChooseConfig(m_data->egl_display, egl_config_attribs, &m_data->egl_config, 1, &m_data->num_configs);
        if (!m_data->success) {
            // TODO: Properly handle this error (requires change to default window
            // API to change return on all window types to bool).
            fprintf(stderr, "Failed to choose config (eglError: %d)\n", eglGetError());
            exit(EXIT_FAILURE);
        }
        if (m_data->num_configs != 1) {
            fprintf(stderr, "Didn't get exactly one config, but %d\n", m_data->num_configs);
            exit(EXIT_FAILURE);
        }

        m_data->success = eglBindAPI(EGL_OPENGL_API);
        if (!m_data->success) {
            // TODO: Properly handle this error (requires change to default window
            // API to change return on all window types to bool).
            fprintf(stderr, "Failed to bind OpenGL API.\n");
            exit(EXIT_FAILURE);
        }

        m_data->egl_context = eglCreateContext(m_data->egl_display, m_data->egl_config, EGL_NO_CONTEXT, NULL);
        if (!m_data->egl_context) {
            fprintf(stderr, "Unable to create EGL context (eglError: %d)\n",eglGetError());
            exit(EXIT_FAILURE);
        }

        m_data->success = eglMakeCurrent(m_data->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, m_data->egl_context);
        if (!m_data->success) {
            fprintf(stderr, "Failed to make context current (eglError: %d)\n", eglGetError());
            exit(EXIT_FAILURE);
        }

        return 0;
    };

    void query() {
        const GLubyte* ven = glGetString(GL_VENDOR);
        printf("GL_VENDOR=%s\n", ven);
        const GLubyte* ren = glGetString(GL_RENDERER);
        printf("GL_RENDERER=%s\n", ren);
        const GLubyte* ver = glGetString(GL_VERSION);
        printf("GL_VERSION=%s\n", ver);
        const GLubyte* sl = glGetString(GL_SHADING_LANGUAGE_VERSION);
        printf("GL_SHADING_LANGUAGE_VERSION=%s\n", sl);
        const GLubyte* gl_ext = glGetString(GL_EXTENSIONS);
        printf("GL_EXTENSIONS=%s\n",gl_ext);
    }

    void release() {
        eglTerminate(m_data->egl_display);
        delete m_data;
        for (int i = 0; i < MAX_NUM_RESOURCES; i++) {
            if (cuda_res[i]) {
                cudaError_t err = cudaGraphicsUnregisterResource(cuda_res[i]);
                if( err != cudaSuccess ) {
                    std::cout << "cudaGraphicsUnregisterResource failed: " << err << std::endl;
                }
            }
        }
    }

    void draw(py::array_t<float> x) {
        //printf("draw\n");
        int size = 3 * m_windowWidth * m_windowHeight;
        //unsigned char *data2 = new unsigned char[size];

        auto ptr = (float *) x.mutable_data();

        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_TRIANGLES);
        glColor3f(1, 0, 0);
        glVertex2f(0,  1);

        glColor3f(0, 1, 0);
        glVertex2f(-1, -1);

        glColor3f(0, 0, 1);
        glVertex2f(1, -1);
        glEnd();

        eglSwapBuffers( m_data->egl_display, m_data->egl_surface);
        glReadPixels(0,0,m_windowWidth,m_windowHeight,GL_RGB, GL_FLOAT, ptr);
        //unsigned error = lodepng::encode("test.png", (unsigned char*)data2, m_windowWidth, m_windowHeight, LCT_RGB, 8);
        //delete data2;
    }

    void draw_py(py::array_t<float> x) {
        /*auto r = x.mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false
            for (ssize_t i = 0; i < r.shape(0); i++)
                for (ssize_t j = 0; j < r.shape(1); j++)
                    for (ssize_t k = 0; k < r.shape(2); k++)
                        r(i, j, k) += 1.0;*/

        std::fill(x.mutable_data(), x.mutable_data() + x.size(), 42);
    }

    void map_tensor(GLuint tid, int width, int height, std::size_t data) {
        cudaError_t err;
        /*
        for (int i=0; i < MAX_NUM_RESOURCES; i++) {
            if (cuda_res[i] != NULL) {
                printf("cuda res: %d not null\n", i);
            }
        }*/
        std::cout << "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT" << std::endl;
        if (cuda_res[tid] == NULL) {
            err = cudaGraphicsGLRegisterImage(&(cuda_res[tid]), tid, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
            if( err != cudaSuccess ) {
                std::cout << "cudaGraphicsGLRegisterImage failed: " << err << std::endl;
            }
        }

        err = cudaGraphicsMapResources(1, &(cuda_res[tid]));
        if( err != cudaSuccess ) {
            std::cout << "cudaGraphicsMapResources failed: " << err << std::endl;
        }

        cudaArray* array;
        err = cudaGraphicsSubResourceGetMappedArray(&array, cuda_res[tid], 0, 0);
        if( err != cudaSuccess ) {
            std::cout << "cudaGraphicsSubResourceGetMappedArray failed: " << err << std::endl;
        }

        // copy data
        err = cudaMemcpy2DFromArray((void*)data, width*4*sizeof(float), array, 0, 0, width*4*sizeof(float), height, cudaMemcpyDeviceToDevice);
        if( err != cudaSuccess ) {
            std::cout << "cudaMemcpy2DFromArray failed: " << err << std::endl;
        }

        err = cudaGraphicsUnmapResources(1, &(cuda_res[tid]));
        if( err != cudaSuccess ) {
            std::cout << "cudaGraphicsUnmapResources failed: " << err << std::endl;
        }
    }
};


PYBIND11_MODULE(CppEGLRenderer, m) {
    py::class_<CppEGLRenderer>(m, "CppEGLRenderer")
        .def(py::init<int, int, int>())
        .def("init", &CppEGLRenderer::init)
        .def("query", &CppEGLRenderer::query)
        .def("map_tensor", &CppEGLRenderer::map_tensor)
        .def("draw", &CppEGLRenderer::draw, py::arg().noconvert())
        .def("release", &CppEGLRenderer::release)
        .def("draw_py", &CppEGLRenderer::draw_py, py::arg().noconvert());

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}

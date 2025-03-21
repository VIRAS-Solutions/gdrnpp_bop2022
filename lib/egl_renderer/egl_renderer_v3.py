"""
ref:
https://github.com/NVlabs/PoseRBPF/tree/master/ycb_render
store model infos in dict instead of list, which allows adding objects dynamically
"""
import ctypes
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
from pprint import pprint
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import OpenGL.GL as GL
import torch
from PIL import Image
import pyassimp
from pyassimp import load, release
from transforms3d.euler import euler2quat, mat2euler, quat2euler
from transforms3d.quaternions import axangle2quat, mat2quat, qinverse, qmult

cur_dir = osp.dirname(osp.abspath(__file__))
# sys.path.insert(0, cur_dir)
from . import CppEGLRenderer
from .glutils.meshutil import (
    homotrans,
    lookat,
    mat2rotmat,
    mat2xyz,
    perspective,
    quat2rotmat,
    safemat2quat,
    xyz2mat,
    loadTexture,
    im2Texture,
    shader_from_path,
    load_mesh_pyassimp,
    load_mesh_sixd,
    get_vertices_extent,
)

from .glutils.egl_offscreen_context import OffscreenContext
from lib.utils import logger


class EGLRenderer(object):
    def __init__(
        self,
        model_paths=None,
        K=None,
        model_ids=None,
        texture_paths=None,
        model_colors=None,
        width=640,
        height=480,
        gpu_id=None,
        render_marker=False,
        vertex_scale=1.0,
        znear=0.25,
        zfar=6.0,
        model_loadfn=None,
        use_cache=False,
        cad_model_colors=None,
    ):
        if model_loadfn == "pyassimp":
            self.model_load_fn = load_mesh_pyassimp
        elif model_loadfn == "pysixd":
            self.model_load_fn = load_mesh_sixd
        else:
            self.model_load_fn = load_mesh_sixd  # default using pysixd .ply loader
        self.use_cache = use_cache

        if gpu_id is None:
            cuda_device_idx = torch.cuda.current_device()
        else:
            with torch.cuda.device(gpu_id):
                cuda_device_idx = torch.cuda.current_device()
        self._context = OffscreenContext(gpu_id=cuda_device_idx)
        self.render_marker = render_marker

        self.texUnitUniform = None
        self.width = width
        self.height = height

        self.znear = znear
        self.zfar = zfar
        self.poses_trans = []
        self.poses_rot = []

        self.r = CppEGLRenderer.CppEGLRenderer(width, height, cuda_device_idx)
        print(self.r.init())
        print(self.r.query())
        quit()
        self.r.init()
        self.glstring = GL.glGetString(GL.GL_VERSION)
        print(self.glstring)
        from OpenGL.GL import shaders

        self.shaders = shaders

        shader_types = {
            "shader_bbox": ("shader_bbox.vs", "shader_bbox.frag"),
            "shader_textureless_texture": (
                "shader_textureless_texture.vs",
                "shader_textureless_texture.frag",
            ),
            "shader_material": ("shader_material.vs", "shader_material.frag"),
            "shader_simple": ("shader_simple.vs", "shader_simple.frag"),
            # "shader_bg": ("background.vs", "background.frag"),
        }
        self.shaders_dict = {}
        for _s_type in shader_types:
            self.shaders_dict[_s_type] = {
                "vertex": self.shaders.compileShader(
                    shader_from_path(shader_types[_s_type][0]),
                    GL.GL_VERTEX_SHADER,
                ),
                "fragment": self.shaders.compileShader(
                    shader_from_path(shader_types[_s_type][1]),
                    GL.GL_FRAGMENT_SHADER,
                ),
            }

        self.shader_programs = {}
        for _s_type in shader_types:
            self.shader_programs[_s_type] = self.shaders.compileProgram(
                self.shaders_dict[_s_type]["vertex"],
                self.shaders_dict[_s_type]["fragment"],
            )
        # self.texUnitUniform = GL.glGetUniformLocation(self.shader_programs['shader'], "uTexture")
        self.texUnitUniform = GL.glGetUniformLocation(self.shader_programs["shader_textureless_texture"], "uTexture")

        self.lightpos = [0, 0, 0]
        self.lightcolor = [1, 1, 1]

        self.fbo = GL.glGenFramebuffers(1)
        self.color_tex = GL.glGenTextures(1)
        self.color_tex_2 = GL.glGenTextures(1)
        self.color_tex_3 = GL.glGenTextures(1)
        self.color_tex_4 = GL.glGenTextures(1)
        self.color_tex_5 = GL.glGenTextures(1)
        self.depth_tex = GL.glGenTextures(1)
        # print("fbo {}, color_tex {}, color_tex_2 {}, color_tex_3 {}, color_tex_4 {}, color_tex_5 {}, depth_tex {}".format(
        #     int(self.fbo), int(self.color_tex), int(self.color_tex_2), int(self.color_tex_3),
        #     int(self.color_tex_4), int(self.color_tex_5), int(self.depth_tex)))

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_2)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_3)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_4)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_tex_5)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            self.width,
            self.height,
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            None,
        )

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.depth_tex)
        GL.glTexImage2D.wrappedOperation(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH24_STENCIL8,
            self.width,
            self.height,
            0,
            GL.GL_DEPTH_STENCIL,
            GL.GL_UNSIGNED_INT_24_8,
            None,
        )

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self.color_tex,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT1,
            GL.GL_TEXTURE_2D,
            self.color_tex_2,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT2,
            GL.GL_TEXTURE_2D,
            self.color_tex_3,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT3,
            GL.GL_TEXTURE_2D,
            self.color_tex_4,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT4,
            GL.GL_TEXTURE_2D,
            self.color_tex_5,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_STENCIL_ATTACHMENT,
            GL.GL_TEXTURE_2D,
            self.depth_tex,
            0,
        )
        GL.glViewport(0, 0, self.width, self.height)
        GL.glDrawBuffers(
            5,
            [
                GL.GL_COLOR_ATTACHMENT0,
                GL.GL_COLOR_ATTACHMENT1,
                GL.GL_COLOR_ATTACHMENT2,
                GL.GL_COLOR_ATTACHMENT3,
                GL.GL_COLOR_ATTACHMENT4,
            ],
        )

        assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE

        self.fov = 20
        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        P = perspective(self.fov, float(self.width) / float(self.height), 0.01, 100)
        V = lookat(self.camera, self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)
        self.P = np.ascontiguousarray(P, np.float32)
        self.grid = self.generate_grid()

        # self.bg_VAO, self.bg_indices = self.set_bg_buffers()
        self.set_camera_default()
        if K is not None:
            self.set_projection_matrix(K, width, height, znear, zfar)

        self.is_rotating = False  # added mouse interaction

        # store model infos (a dict of dicts)
        # model_id:
        #   model_path
        #   vertices, faces,
        #   seg_color: # for per-object instance seg
        #   cad_model_color: for cad models
        #   materials (a list, single or multiple),
        #   VAOs(a list, single or multiple),
        #   VBOs(a list, single or multiple),
        #   texture
        #   is_cad, is_textured, is_materialed
        self.models = {}
        if model_paths is not None:
            self.load_objects(
                model_paths,
                texture_paths,
                model_colors,
                model_ids=model_ids,
                vertex_scale=vertex_scale,
                cad_model_colors=cad_model_colors,
            )

    def generate_grid(self):
        VAO = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(VAO)
        vertexData = []
        for i in np.arange(-1, 1, 0.05):  # 160
            vertexData.append([i, 0, -1, 0, 0, 0, 0, 0])
            vertexData.append([i, 0, 1, 0, 0, 0, 0, 0])
            vertexData.append([1, 0, i, 0, 0, 0, 0, 0])
            vertexData.append([-1, 0, i, 0, 0, 0, 0, 0])
        vertexData = np.array(vertexData).astype(np.float32) * 3
        # Need VBO for triangle vertices and texture UV coordinates
        VBO = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            vertexData.nbytes,
            vertexData,
            GL.GL_STATIC_DRAW,
        )

        # enable array and set up data
        positionAttrib = GL.glGetAttribLocation(self.shader_programs["shader_simple"], "aPosition")
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 8 * 4, None)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        return VAO

    # def set_bg_buffers(self):
    #     # TODO: make it work
    #     # Set up background render quad in NDC

    #     # yapf: disable
    #     # quad = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    #     # tex = [[0, 1], [1, 1], [1, 0], [0, 0]]
    #     quad = [[-1, -1], [-1, 1], [1, 1], [1, -1]]
    #     tex  = [[ 0,  0], [ 0, 1], [1, 1], [1,  0]]
    #     # yapf: enable
    #     vertices = np.array(quad, dtype=np.float32)
    #     texcoord = np.array(tex, dtype=np.float32)
    #     vertexData = np.concatenate([vertices, texcoord], axis=-1).astype(np.float32)
    #     # indices = np.array([0, 1, 2, 0, 2, 3], np.int32)
    #     indices = np.array([0, 1, 3, 0, 2, 3], np.int32)

    #     VAO = GL.glGenVertexArrays(1)
    #     GL.glBindVertexArray(VAO)
    #     # Need VBO for triangle vertices and texture UV coordinates
    #     VBO = GL.glGenBuffers(1)
    #     GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
    #     GL.glBufferData(GL.GL_ARRAY_BUFFER, vertexData.nbytes, vertexData, GL.GL_STATIC_DRAW)

    #     # enable array and set up data
    #     _shader_type = "shader_bg"
    #     positionAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aPosition")
    #     coordsAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aTexcoord")

    #     GL.glEnableVertexAttribArray(0)
    #     GL.glEnableVertexAttribArray(1)
    #     # index, size, type, normalized, stride=vertexData.shape[1]*4, pointer
    #     GL.glVertexAttribPointer(positionAttrib, 2, GL.GL_FLOAT, GL.GL_FALSE, 4*4, None)  # 0
    #     GL.glVertexAttribPointer(coordsAttrib, 2, GL.GL_FLOAT, GL.GL_TRUE, 4*4, ctypes.c_void_p(2*4)) # 2*4=8

    #     GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    #     GL.glBindVertexArray(0)

    #     return VAO, indices

    def extent_to_bbox3d(self, xsize, ysize, zsize, is_gt=False):
        # yapf: disable
        bb = np.asarray([[-xsize / 2,  ysize / 2,  zsize / 2],
                         [ xsize / 2,  ysize / 2,  zsize / 2],
                         [-xsize / 2, -ysize / 2,  zsize / 2],
                         [ xsize / 2, -ysize / 2,  zsize / 2],
                         [-xsize / 2,  ysize / 2, -zsize / 2],
                         [ xsize / 2,  ysize / 2, -zsize / 2],
                         [-xsize / 2, -ysize / 2, -zsize / 2],
                         [ xsize / 2, -ysize / 2, -zsize / 2]])
        # Set up rendering data
        if is_gt:
            colors = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]]
        else:
            colors = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]
        # yapf: enable
        """
            0 -------- 1
           /|         /|
          2 -------- 3 .
          | |        | |
          . 4 -------- 5
          |/         |/
          6 -------- 7
        """
        indices = [
            0,
            1,
            0,
            2,
            3,
            1,
            3,
            2,
            4,
            5,
            4,
            6,
            7,
            5,
            7,
            6,
            0,
            4,
            1,
            5,
            2,
            6,
            3,
            7,
        ]
        indices = np.array(indices, dtype=np.int32)

        vertices = np.array(bb, dtype=np.float32)
        normals = np.zeros_like(vertices)
        colors = np.array(colors, dtype=np.float32)
        vertexData = np.concatenate([vertices, normals, colors], axis=-1).astype(np.float32)

        VAO = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(VAO)
        # Need VBO for triangle vertices and texture UV coordinates
        VBO = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            vertexData.nbytes,
            vertexData,
            GL.GL_STATIC_DRAW,
        )

        # enable array and set up data
        _shader_type = "shader_bbox"
        positionAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aPosition")
        # normalAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aNormal")
        colorAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aColor")

        GL.glEnableVertexAttribArray(0)
        GL.glEnableVertexAttribArray(2)
        # index, size, type, normalized, stride=vertexData.shape[1]*4, pointer
        GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 9 * 4, None)  # 0
        GL.glVertexAttribPointer(
            colorAttrib,
            3,
            GL.GL_FLOAT,
            GL.GL_FALSE,
            9 * 4,
            ctypes.c_void_p(6 * 4),
        )  # 6*4=24

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        return VAO, indices

    def load_object(
        self,
        obj_path,
        obj_id=None,
        texture_path="",
        vertex_scale=1.0,
        model_color=np.array([10.0, 10.0, 10.0]) / 255.0,
        cad_model_color=None,
    ):
        assert osp.exists(obj_path), obj_path
        if obj_id is None:
            obj_id = len(self.models)
        res_model = {
            "model_path": obj_path,
            "cad_model_color": cad_model_color,
            "seg_color": model_color,
        }

        is_textured = False
        is_materialed = False
        is_cad = False
        if texture_path != "":
            is_textured = True
            logger.info("texture path: {}".format(texture_path))
            texture = loadTexture(texture_path)
            res_model["texture"] = texture
        res_model["is_textured"] = is_textured
        if obj_path.endswith("DAE"):
            is_materialed = True
            vertices, faces, materials = self.load_robot_mesh(obj_path)  # return list of vertices, faces, materials
            res_model["vertices"] = vertices
            res_model["faces"] = faces
            res_model["materials"] = materials
            res_model["texture"] = ""  # dummy
        res_model["is_materialed"] = is_materialed
        if is_materialed:
            _VAOs, _VBOs = [], [], [], []
            for idx in range(len(vertices)):
                vertexData = vertices[idx].astype(np.float32)
                VAO = GL.glGenVertexArrays(1)
                GL.glBindVertexArray(VAO)

                # Need VBO for triangle vertices and texture UV coordinates
                VBO = GL.glGenBuffers(1)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
                GL.glBufferData(
                    GL.GL_ARRAY_BUFFER,
                    vertexData.nbytes,
                    vertexData,
                    GL.GL_STATIC_DRAW,
                )
                positionAttrib = GL.glGetAttribLocation(self.shader_programs["shader_material"], "aPosition")
                normalAttrib = GL.glGetAttribLocation(self.shader_programs["shader_material"], "aNormal")

                GL.glEnableVertexAttribArray(0)
                GL.glEnableVertexAttribArray(1)

                GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, None)
                GL.glVertexAttribPointer(
                    normalAttrib,
                    3,
                    GL.GL_FLOAT,
                    GL.GL_FALSE,
                    24,
                    ctypes.c_void_p(12),
                )

                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
                GL.glBindVertexArray(0)
                _VAOs.append(VAO)
                _VBOs.append(VBO)

            res_model["VAOs"] = _VAOs
            res_model["VBOs"] = _VBOs

        else:
            _shader_type = "shader_textureless_texture"
            logger.info(obj_path)
            mesh = self.model_load_fn(
                obj_path,
                vertex_scale=vertex_scale,
                is_textured=is_textured,
                use_cache=self.use_cache,
                cad_model_color=cad_model_color,
            )
            is_cad = mesh["is_cad"]
            logger.info("is_textured: {} | is_cad: {} | is_materialed: {}".format(is_textured, is_cad, is_materialed))
            # pprint(mesh)
            # check materials
            logger.info("{}".format(list(mesh.keys())))
            mat_diffuse, mat_specular, mat_ambient, mat_shininess = [
                mesh[_k]
                for _k in [
                    "uMatDiffuse",
                    "uMatSpecular",
                    "uMatAmbient",
                    "uMatShininess",
                ]
            ]

            res_model["materials"] = [np.hstack([mat_diffuse, mat_specular, mat_ambient, mat_shininess])]
            res_model["faces"] = faces = mesh["faces"]
            res_model["vertices"] = mesh["vertices"]

            logger.info("colors: {}".format(mesh["colors"].max()))
            vertices = np.concatenate(
                [
                    mesh["vertices"],
                    mesh["normals"],
                    mesh["colors"],
                    mesh["texturecoords"],
                ],
                axis=-1,
            )  # ply models

            vertexData = vertices.astype(np.float32)
            # print(vertexData.shape, faces.shape) #..x8, ..x3
            VAO = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(VAO)

            # Need VBO for triangle vertices and texture UV coordinates
            VBO = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, VBO)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                vertexData.nbytes,
                vertexData,
                GL.GL_STATIC_DRAW,
            )

            # enable array and set up data
            positionAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aPosition")
            normalAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aNormal")
            colorAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aColor")
            coordsAttrib = GL.glGetAttribLocation(self.shader_programs[_shader_type], "aTexcoord")

            GL.glEnableVertexAttribArray(0)
            GL.glEnableVertexAttribArray(1)
            GL.glEnableVertexAttribArray(2)
            GL.glEnableVertexAttribArray(3)  # added

            GL.glVertexAttribPointer(positionAttrib, 3, GL.GL_FLOAT, GL.GL_FALSE, 11 * 4, None)  # 0
            GL.glVertexAttribPointer(
                normalAttrib,
                3,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                11 * 4,
                ctypes.c_void_p(3 * 4),
            )  # 3*4=12
            GL.glVertexAttribPointer(
                colorAttrib,
                3,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                11 * 4,
                ctypes.c_void_p(6 * 4),
            )  # 6*4=24
            GL.glVertexAttribPointer(
                coordsAttrib,
                2,
                GL.GL_FLOAT,
                GL.GL_TRUE,
                11 * 4,
                ctypes.c_void_p(9 * 4),
            )  # 9*4 = 36

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindVertexArray(0)

            res_model["VAOs"] = [VAO]
            res_model["VBOs"] = [VBO]

        self.poses_rot.append(np.eye(4))
        self.poses_trans.append(np.eye(4))
        res_model["is_cad"] = is_cad
        self.models[obj_id] = res_model

    def load_robot_mesh(self, collada_path):
        # load collada file and return vertices, faces, materials
        mesh_file = collada_path.strip().split("/")[-1]  # for offset the robot mesh
        scene = load(collada_path)  # load collada
        offset = self._offset_map[mesh_file]
        return self.recursive_load(scene.rootnode, [], [], [], offset)

    def recursive_load(self, node, vertices, faces, materials, offset):
        if node.meshes:
            transform = node.transformation
            for idx, mesh in enumerate(node.meshes):
                # pprint(vars(mesh))
                if mesh.faces.shape[-1] != 3:  # ignore boundLineSet
                    continue
                mat = mesh.material
                pprint(vars(mat))
                mat_diffuse = np.array(mat.properties["diffuse"])[:3]

                if "specular" in mat.properties:
                    mat_specular = np.array(mat.properties["specular"])[:3]
                else:
                    mat_specular = [0.5, 0.5, 0.5]
                    mat_diffuse = [0.8, 0.8, 0.8]

                if "ambient" in mat.properties:
                    mat_ambient = np.array(mat.properties["ambient"])[:3]  # phong shader
                else:
                    mat_ambient = [0, 0, 0]

                if "shininess" in mat.properties:
                    mat_shininess = max(mat.properties["shininess"], 1)  # avoid the 0 shininess
                else:
                    mat_shininess = 1

                mesh_vertex = homotrans(transform, mesh.vertices) - offset  # subtract the offset
                mesh_normals = transform[:3, :3].dot(mesh.normals.transpose()).transpose()  # normal stays the same
                vertices.append(np.concatenate([mesh_vertex, mesh_normals], axis=-1))
                faces.append(mesh.faces)
                materials.append(np.hstack([mat_diffuse, mat_specular, mat_ambient, mat_shininess]))
                # concat speed, render speed, bind & unbind, memory
        for child in node.children:
            self.recursive_load(child, vertices, faces, materials, offset)
        return vertices, faces, materials

    def load_objects(
        self,
        model_paths,
        texture_paths=None,
        model_colors=[[0.9, 0, 0], [0.6, 0, 0], [0.3, 0, 0]],
        model_ids=None,
        vertex_scale=1.0,
        cad_model_colors=None,
    ):
        if model_ids is not None:
            assert len(model_ids) == len(model_paths)
        else:
            model_ids = [i for i in range(len(model_paths))]  # ids default start from 0
        self.models.update({_id: {} for _id in model_ids})

        if model_colors is None:  # init render stuff
            class_colors_all = [((x + 1) * 10, (x + 1) * 10, (x + 1) * 10) for x in range(len(model_paths))]
            model_colors = [np.array(class_colors_all[i]) / 255.0 for i in range(len(model_paths))]
        if texture_paths is None:
            texture_paths = ["" for i in range(len(model_paths))]
        if cad_model_colors is not None:
            assert len(cad_model_colors) == len(model_paths)
        else:
            cad_model_colors = [None for _ in model_paths]

        for i in tqdm(range(len(model_paths))):
            self.load_object(
                model_paths[i],
                obj_id=model_ids[i],
                texture_path=texture_paths[i],
                vertex_scale=vertex_scale,
                model_color=model_colors[i],
                cad_model_color=cad_model_colors[i],
            )

    def set_camera(self, camera, target, up):
        self.camera = camera
        self.target = target
        self.up = up
        V = lookat(self.camera, self.target, up=self.up)

        self.V = np.ascontiguousarray(V, np.float32)

    def set_camera_default(self):
        self.V = np.eye(4)

    def set_fov(self, fov):
        self.fov = fov
        # this is vertical fov (fovy)
        P = perspective(self.fov, float(self.width) / float(self.height), 0.01, 100)
        self.P = np.ascontiguousarray(P, np.float32)

    def set_projection_matrix(self, K, width, height, znear, zfar):
        """
        set projection matrix according to real camera intrinsics
        P.T = [
            [2*fx/w,     0,           0,  0],
            [0,          -2*fy/h,     0,  0],
            [(2*px-w)/w, (-2*py+h)/h, -q, 1],
            [0,          0,           qn, 0],
        ]
        sometimes P.T[2,:] *= -1, P.T[1, :] *= -1
        """
        fx = K[0, 0]
        fy = K[1, 1]
        px = K[0, 2]
        py = K[1, 2]
        fc = zfar
        nc = znear
        q = -(fc + nc) / float(fc - nc)
        qn = -2 * (fc * nc) / float(fc - nc)

        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = +2 * fx / width
        P[1, 1] = -2 * fy / height
        P[2, 0] = (+2 * px - width) / width
        P[2, 1] = (-2 * py + height) / height
        P[2, 2] = -q
        P[2, 3] = 1.0
        P[3, 2] = qn
        self.P = P

    def set_light_color(self, color):
        self.lightcolor = color

    def draw_bg(self, im):
        texture_id = im2Texture(im, flip_v=True)
        # draw texture
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glBegin(GL.GL_QUADS)
        # yapf: disable
        GL.glTexCoord2f(0, 0)
        GL.glVertex2f(-1, -1)
        GL.glTexCoord2f(0, 1)
        GL.glVertex2f(-1, 1)
        GL.glTexCoord2f(1, 1)
        GL.glVertex2f(1, 1)
        GL.glTexCoord2f(1, 0)
        GL.glVertex2f(1, -1)
        # yapf: enable
        GL.glEnd()
        GL.glDisable(GL.GL_TEXTURE_2D)
        # GL.glBindVertexArray(0)
        # GL.glUseProgram(0)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)  # clear depth
        # GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # clear depth
        GL.glEnable(GL.GL_DEPTH_TEST)

        # _shader_type = 'shader_bg'
        # shader = self.shader_programs[_shader_type]

        # GL.glEnable(GL.GL_TEXTURE_2D)
        # GL.glBegin(GL.GL_QUADS)
        # GL.glUseProgram(shader)
        # # whether fixed-point data values should be normalized ( GL_TRUE ) or converted directly as fixed-point values ( GL_FALSE )
        # try:
        #     GL.glActiveTexture(GL.GL_TEXTURE0)  # Activate texture
        #     GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        #     # GL.glUniform1i(self.texUnitUniform, 0)
        #     GL.glUniform1i(GL.glGetUniformLocation(shader, "uTexture"), 0)
        #     GL.glBindVertexArray(self.bg_VAO) # Activate array
        #     # draw triangles
        #     GL.glDrawElements(GL.GL_TRIANGLES, len(self.bg_indices), GL.GL_UNSIGNED_INT, self.bg_indices)
        # except:
        #     logger.warn('err in draw bg')
        # finally:
        #     GL.glEnd()
        #     GL.glDisable(GL.GL_TEXTURE_2D)

        #     GL.glBindVertexArray(0)
        #     GL.glUseProgram(0)
        #     GL.glClear(GL.GL_DEPTH_BUFFER_BIT)  # clear depth

    def render(
        self,
        obj_ids,
        poses,
        K=None,
        to_bgr=True,
        to_255=True,
        rot_type="mat",
        instance_colors=None,
        light_pos=None,
        light_color=None,
        image_tensor=None,
        seg_tensor=None,
        normal_tensor=None,
        pc_obj_tensor=None,
        pc_cam_tensor=None,
        phong={"ambient": 0.4, "diffuse": 0.8, "specular": 0.3},
        extents=None,
        gt_extents=None,
        background=None,
    ):
        # get un-occluded instance masks by rendering one by one
        if isinstance(obj_ids, int):
            obj_ids = [obj_ids]
        if isinstance(poses, np.ndarray):
            poses = [poses]
        if K is not None:
            if image_tensor is not None:
                if self.height != image_tensor.shape[0] or self.width != image_tensor.shape[1]:
                    self.height = image_tensor.shape[0]
                    self.width = image_tensor.shape[1]
            self.set_projection_matrix(
                K,
                width=self.width,
                height=self.height,
                znear=self.znear,
                zfar=self.zfar,
            )
        if light_pos is not None:
            self.set_light_pos(light_pos)
        if light_color is not None:
            self.set_light_color(light_color)
        if instance_colors is not None:
            assert len(instance_colors) == len(obj_ids)
        else:
            instance_colors = [self.models[obj_id]["seg_color"] for obj_id in obj_ids]
        if extents is not None:
            assert len(extents) == len(obj_ids)
        if gt_extents is not None:
            assert len(gt_extents) == len(obj_ids)
        self.set_poses(poses, rot_type=rot_type)

        # self.lightpos = np.random.uniform(-1, 1, 3)
        # frame = 0
        GL.glClearColor(0, 0, 0, 1)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        # GL.glLightModeli(GL.GL_LIGHT_MODEL_TWO_SIDE, GL.GL_TRUE)
        if background is not None:
            self.draw_bg(background)

        if self.render_marker:
            # render some grid and directions
            GL.glUseProgram(self.shader_programs["shader_simple"])
            GL.glBindVertexArray(self.grid)
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self.shader_programs["shader_simple"], "V"),
                1,
                GL.GL_TRUE,
                self.V,
            )
            GL.glUniformMatrix4fv(
                GL.glGetUniformLocation(self.shader_programs["shader_simple"], "uProj"),
                1,
                GL.GL_FALSE,
                self.P,
            )
            GL.glDrawElements(
                GL.GL_LINES,
                160,
                GL.GL_UNSIGNED_INT,
                np.arange(160, dtype=np.int),
            )
            GL.glBindVertexArray(0)
            GL.glUseProgram(0)
            # end rendering markers

        # render 3d bboxes ================================================================================
        if extents is not None:
            thickness = 1.5
            GL.glLineWidth(thickness)
            _shader_name = "shader_bbox"
            shader = self.shader_programs[_shader_name]
            for i, extent in enumerate(extents):
                GL.glUseProgram(shader)
                _vertexData, _indices = self.extent_to_bbox3d(extent[0], extent[1], extent[2], is_gt=False)
                GL.glBindVertexArray(_vertexData)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader, "V"), 1, GL.GL_TRUE, self.V)
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "uProj"),
                    1,
                    GL.GL_FALSE,
                    self.P,
                )
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "pose_trans"),
                    1,
                    GL.GL_FALSE,
                    self.poses_trans[i],
                )
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "pose_rot"),
                    1,
                    GL.GL_TRUE,
                    self.poses_rot[i],
                )

                GL.glDrawElements(GL.GL_LINES, len(_indices), GL.GL_UNSIGNED_INT, _indices)
                GL.glBindVertexArray(0)
                GL.glUseProgram(0)
            GL.glLineWidth(1.0)

            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)  # clear depth of 3d bboxes

        if gt_extents is not None:
            thickness = 1.5
            GL.glLineWidth(thickness)
            _shader_name = "shader_bbox"
            shader = self.shader_programs[_shader_name]
            for i, gt_extent in enumerate(gt_extents):
                GL.glUseProgram(shader)
                _vertexData, _indices = self.extent_to_bbox3d(gt_extent[0], gt_extent[1], gt_extent[2], is_gt=True)
                GL.glBindVertexArray(_vertexData)
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader, "V"), 1, GL.GL_TRUE, self.V)
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "uProj"),
                    1,
                    GL.GL_FALSE,
                    self.P,
                )
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "pose_trans"),
                    1,
                    GL.GL_FALSE,
                    self.poses_trans[i],
                )
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "pose_rot"),
                    1,
                    GL.GL_TRUE,
                    self.poses_rot[i],
                )

                GL.glDrawElements(GL.GL_LINES, len(_indices), GL.GL_UNSIGNED_INT, _indices)
                GL.glBindVertexArray(0)
                GL.glUseProgram(0)
            GL.glLineWidth(1.0)
            GL.glClear(GL.GL_DEPTH_BUFFER_BIT)  # clear depth of 3d bboxes
        # size = 0
        for i in range(len(obj_ids)):  ##################################
            obj_id = obj_ids[i]
            cur_model = self.models[obj_id]
            is_textured = cur_model["is_textured"]
            is_materialed = cur_model["is_materialed"]
            # active shader program
            if is_materialed:  # for mesh in the robot mesh
                num = len(cur_model["materials"])
                for idx in range(num):
                    # the materials stored in vertex attribute instead of uniforms to avoid bind & unbind
                    shader = self.shader_programs["shader_material"]
                    GL.glUseProgram(shader)
                    GL.glUniformMatrix4fv(
                        GL.glGetUniformLocation(shader, "V"),
                        1,
                        GL.GL_TRUE,
                        self.V,
                    )
                    GL.glUniformMatrix4fv(
                        GL.glGetUniformLocation(shader, "uProj"),
                        1,
                        GL.GL_FALSE,
                        self.P,
                    )
                    GL.glUniformMatrix4fv(
                        GL.glGetUniformLocation(shader, "pose_trans"),
                        1,
                        GL.GL_FALSE,
                        self.poses_trans[i],
                    )
                    GL.glUniformMatrix4fv(
                        GL.glGetUniformLocation(shader, "pose_rot"),
                        1,
                        GL.GL_TRUE,
                        self.poses_rot[i],
                    )
                    GL.glUniform3f(
                        GL.glGetUniformLocation(shader, "uLightPosition"),
                        *self.lightpos,
                    )
                    GL.glUniform3f(
                        GL.glGetUniformLocation(shader, "instance_color"),
                        *instance_colors[i],
                    )
                    GL.glUniform3f(
                        GL.glGetUniformLocation(shader, "uLightColor"),
                        *self.lightcolor,
                    )
                    GL.glUniform3f(
                        GL.glGetUniformLocation(shader, "uMatDiffuse"),
                        *cur_model["materials"][idx][:3],
                    )
                    GL.glUniform3f(
                        GL.glGetUniformLocation(shader, "uMatSpecular"),
                        *cur_model["materials"][idx][3:6],
                    )
                    GL.glUniform3f(
                        GL.glGetUniformLocation(shader, "uMatAmbient"),
                        *cur_model["materials"][idx][6:9],
                    )
                    GL.glUniform1f(
                        GL.glGetUniformLocation(shader, "uMatShininess"),
                        cur_model["materials"][idx][-1],
                    )

                    GL.glUniform1f(
                        GL.glGetUniformLocation(shader, "uLightAmbientWeight"),
                        phong["ambient"],
                    )
                    GL.glUniform1f(
                        GL.glGetUniformLocation(shader, "uLightDiffuseWeight"),
                        phong["diffuse"],
                    )
                    GL.glUniform1f(
                        GL.glGetUniformLocation(shader, "uLightSpecularWeight"),
                        phong["specular"],
                    )

                    try:
                        GL.glBindVertexArray(cur_model["VAOs"][idx])
                        GL.glDrawElements(
                            GL.GL_TRIANGLES,
                            cur_model["faces"][idx].size,
                            GL.GL_UNSIGNED_INT,
                            cur_model["faces"][idx],
                        )
                    finally:
                        GL.glBindVertexArray(0)
                        GL.glUseProgram(0)
            else:  # is_textured / is_cad / is_colored #################################################################
                _shader_type = "shader_textureless_texture"
                shader = self.shader_programs[_shader_type]

                GL.glUseProgram(shader)
                # whether fixed-point data values should be normalized ( GL_TRUE ) or converted directly as fixed-point values ( GL_FALSE )
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(shader, "V"), 1, GL.GL_TRUE, self.V)
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "uProj"),
                    1,
                    GL.GL_FALSE,
                    self.P,
                )
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "pose_trans"),
                    1,
                    GL.GL_FALSE,
                    self.poses_trans[i],
                )
                GL.glUniformMatrix4fv(
                    GL.glGetUniformLocation(shader, "pose_rot"),
                    1,
                    GL.GL_TRUE,
                    self.poses_rot[i],
                )

                GL.glUniform3f(
                    GL.glGetUniformLocation(shader, "uLightPosition"),
                    *self.lightpos,
                )
                GL.glUniform3f(
                    GL.glGetUniformLocation(shader, "instance_color"),
                    *instance_colors[i],
                )
                GL.glUniform3f(
                    GL.glGetUniformLocation(shader, "uLightColor"),
                    *self.lightcolor,
                )
                GL.glUniform1i(
                    GL.glGetUniformLocation(shader, "uUseTexture"),
                    int(is_textured),
                )

                GL.glUniform1f(
                    GL.glGetUniformLocation(shader, "uLightAmbientWeight"),
                    phong["ambient"],
                )
                GL.glUniform1f(
                    GL.glGetUniformLocation(shader, "uLightDiffuseWeight"),
                    phong["diffuse"],
                )
                GL.glUniform1f(
                    GL.glGetUniformLocation(shader, "uLightSpecularWeight"),
                    phong["specular"],
                )

                try:
                    if is_textured:
                        GL.glActiveTexture(GL.GL_TEXTURE0)  # Activate texture
                        GL.glBindTexture(GL.GL_TEXTURE_2D, cur_model["texture"])
                        # GL.glUniform1i(self.texUnitUniform, 0)
                        GL.glUniform1i(GL.glGetUniformLocation(shader, "uTexture"), 0)
                        GL.glUniform3f(
                            GL.glGetUniformLocation(shader, "uMatDiffuse"),
                            *cur_model["materials"][0][:3],
                        )
                        GL.glUniform3f(
                            GL.glGetUniformLocation(shader, "uMatSpecular"),
                            *cur_model["materials"][0][3:6],
                        )
                        GL.glUniform3f(
                            GL.glGetUniformLocation(shader, "uMatAmbient"),
                            *cur_model["materials"][0][6:9],
                        )
                        GL.glUniform1f(
                            GL.glGetUniformLocation(shader, "uMatShininess"),
                            cur_model["materials"][0][-1],
                        )
                    GL.glBindVertexArray(cur_model["VAOs"][0])  # Activate array
                    # draw triangles
                    GL.glDrawElements(
                        GL.GL_TRIANGLES,
                        cur_model["faces"].size,
                        GL.GL_UNSIGNED_INT,
                        cur_model["faces"],
                    )
                except:
                    logger.warn("err in render")
                finally:
                    GL.glBindVertexArray(0)
                    GL.glUseProgram(0)

        # draw done

        GL.glDisable(GL.GL_DEPTH_TEST)
        # mapping
        # print('color_tex: {} seg_tex: {}'.format(int(self.color_tex), int(self.color_tex_3)))  # 1, 3
        if image_tensor is not None:
            self.r.map_tensor(
                int(self.color_tex),
                int(self.width),
                int(self.height),
                image_tensor.data_ptr(),
            )
            image_tensor.data = torch.flip(image_tensor, (0,))
            if to_bgr:
                image_tensor.data[:, :, :3] = image_tensor.data[:, :, [2, 1, 0]]
            if to_255:
                image_tensor.data = image_tensor.data * 255
        if seg_tensor is not None:
            self.r.map_tensor(
                int(self.color_tex_3),
                int(self.width),
                int(self.height),
                seg_tensor.data_ptr(),
            )
            seg_tensor.data = torch.flip(seg_tensor, (0,))
        # print(np.unique(seg_tensor.cpu().numpy()))
        if normal_tensor is not None:
            self.r.map_tensor(
                int(self.color_tex_2),
                int(self.width),
                int(self.height),
                normal_tensor.data_ptr(),
            )
        if pc_obj_tensor is not None:
            self.r.map_tensor(
                int(self.color_tex_4),
                int(self.width),
                int(self.height),
                pc_obj_tensor.data_ptr(),
            )
            pc_obj_tensor.data = torch.flip(pc_obj_tensor, (0,))
        if pc_cam_tensor is not None:
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
            print(self.color_tex_5)
            print(self.width)
            print(self.height)
            print(pc_cam_tensor.cpu().numpy().shape)
            print(self.glstring)
            print(self.r)
            self.r.map_tensor(
                int(self.color_tex_5),
                int(self.width),
                int(self.height),
                pc_cam_tensor.data_ptr(),
            )
            pc_cam_tensor.data = torch.flip(pc_cam_tensor, (0,))
            # depth is pc_cam_tensor[:,:,2]
        """
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        frame = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #frame = np.frombuffer(frame,dtype = np.float32).reshape(self.width, self.height, 4)
        frame = frame.reshape(self.height, self.width, 4)[::-1, :]

        # GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT1)
        #normal = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #normal = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)
        #normal = normal[::-1, ]

        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT2)
        seg = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_BGRA, GL.GL_FLOAT)
        #seg = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)
        seg = seg.reshape(self.height, self.width, 4)[::-1, :]

        #pc = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT)
        # seg = np.frombuffer(frame, dtype=np.uint8).reshape(self.width, self.height, 4)

        #pc = np.stack([pc,pc, pc, np.ones(pc.shape)], axis = -1)
        #pc = pc[::-1, ]
        #pc = (1-pc) * 10

        # points in object coordinate
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT3)
        pc2 = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
        pc2 = pc2.reshape(self.height, self.width, 4)[::-1, :]
        pc2 = pc2[:,:,:3]

        # points in camera coordinate
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT4)
        pc3 = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_FLOAT)
        pc3 = pc3.reshape(self.height, self.width, 4)[::-1, :]
        pc3 = pc3[:,:,:3]

        return [frame, seg, pc2, pc3]
        """

    def set_light_pos(self, light):
        self.lightpos = light

    def set_poses(self, poses, rot_type="mat"):
        assert rot_type in ["mat", "quat"], rot_type
        if rot_type == "quat":
            self.poses_rot = [np.ascontiguousarray(quat2rotmat(item[:4])) for item in poses]
            self.poses_trans = [np.ascontiguousarray(xyz2mat(item[4:7])) for item in poses]
        elif rot_type == "mat":
            self.poses_rot = [np.ascontiguousarray(mat2rotmat(item[:3, :3])) for item in poses]
            self.poses_trans = [np.ascontiguousarray(xyz2mat(item[:3, 3])) for item in poses]
        else:
            raise ValueError("wrong rot_type: {}".format(rot_type))

    def close(self):
        # logger.info(self.glstring)
        self.clean()
        self._context.close()
        # TODO: handle errors
        self.r.release()

    def remove_object(self, obj_id):
        GL.glDeleteBuffers(len(self.models[obj_id]["VAOs"]), self.models[obj_id]["VAOs"])
        GL.glDeleteBuffers(len(self.models[obj_id]["VBOs"]), self.models[obj_id]["VBOs"])

        if "texture" in self.models[obj_id] and self.models[obj_id]["texture"] != "":
            GL.glDeleteTextures([self.models[obj_id]["texture"]])

        del self.models[obj_id]
        # self.poses_trans = []  # GC should free things here
        # self.poses_rot = []  # GC should free things here

    def clean(self):
        GL.glDeleteTextures(
            [
                self.color_tex,
                self.color_tex_2,
                self.color_tex_3,
                self.color_tex_4,
                self.depth_tex,
            ]
        )
        self.color_tex = None
        self.color_tex_2 = None
        self.color_tex_3 = None
        self.color_tex_4 = None

        self.depth_tex = None
        GL.glDeleteFramebuffers(1, [self.fbo])
        self.fbo = None
        # TODO: check them
        for obj_id in self.models.keys():
            GL.glDeleteBuffers(len(self.models[obj_id]["VAOs"]), self.models[obj_id]["VAOs"])
            GL.glDeleteBuffers(len(self.models[obj_id]["VBOs"]), self.models[obj_id]["VBOs"])
            if "texture" in self.models[obj_id] and self.models[obj_id]["texture"] != "":
                GL.glDeleteTextures([self.models[obj_id]["texture"]])

        self.models = {}
        self.poses_trans = []  # GC should free things here
        self.poses_rot = []  # GC should free things here


def get_img_model_points_with_coords2d(mask_pred, xyz_pred, coord2d, im_H, im_W, max_num_points=-1, mask_thr=0.5):
    """
    from predicted crop_and_resized xyz, bbox top-left,
    get 2D-3D correspondences (image points, 3D model points)
    Args:
        mask_pred: HW, predicted mask in roi_size
        xyz_pred: HWC, predicted xyz in roi_size(eg. 64)
        coord2d: HW2 coords 2d in roi size
        im_H, im_W
        extent: size of x,y,z
    """
    coord2d = coord2d.copy()
    coord2d[:, :, 0] = coord2d[:, :, 0] * im_W
    coord2d[:, :, 1] = coord2d[:, :, 1] * im_H

    sel_mask = (
        (mask_pred > mask_thr)
        & (abs(xyz_pred[:, :, 0]) > 0.0001)
        & (abs(xyz_pred[:, :, 1]) > 0.0001)
        & (abs(xyz_pred[:, :, 2]) > 0.0001)
    )
    model_points = xyz_pred[sel_mask].reshape(-1, 3)
    image_points = coord2d[sel_mask].reshape(-1, 2)

    if max_num_points >= 4:
        num_points = len(image_points)
        max_keep = min(max_num_points, num_points)
        indices = [i for i in range(num_points)]
        random.shuffle(indices)
        model_points = model_points[indices[:max_keep]]
        image_points = image_points[indices[:max_keep]]
    return image_points, model_points


if __name__ == "__main__":
    # python -m lib.egl_renderer.egl_renderer_v3
    import random
    import glob
    import time
    from tqdm import tqdm
    from transforms3d.axangles import axangle2mat
    import matplotlib.pyplot as plt

    from lib.vis_utils.image import vis_image_mask_bbox_cv2
    from lib.pysixd import inout, misc
    from lib.pysixd.pose_error import calc_rt_dist_m
    from lib.vis_utils.image import grid_show
    from core.utils.utils import get_emb_show
    from core.utils.data_utils import get_2d_coord_np

    random.seed(0)
    # test_ycb_render()
    # exit(0)

    width = 640
    height = 480
    znear = 0.25
    zfar = 6.0
    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
    idx2class = {
        1: "ape",
        2: "benchvise",
        3: "bowl",
        4: "camera",
        5: "can",
        6: "cat",
        7: "cup",
        8: "driller",
        9: "duck",
        10: "eggbox",
        11: "glue",
        12: "holepuncher",
        13: "iron",
        14: "lamp",
        15: "phone",
    }
    classes = idx2class.values()
    classes = sorted(classes)

    model_root = "/gdrnpp_bop2022/datasets/BOP_DATASETS/lm/models/"
    model_paths = [osp.join(model_root, "obj_{:06d}.ply".format(cls_idx)) for cls_idx in idx2class]
    models = [inout.load_ply(model_path, vertex_scale=0.001) for model_path in model_paths]
    extents = [get_vertices_extent(model["pts"]) for model in models]

    coord2d = get_2d_coord_np(width=width, height=height, fmt="HWC")

    renderer = EGLRenderer(
        model_paths,
        K=K,
        width=width,
        height=height,
        render_marker=False,
        vertex_scale=0.001,
        use_cache=True,
    )
    tensor_kwargs = {"device": torch.device("cuda"), "dtype": torch.float32}
    image_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
    seg_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()

    instance_mask_tensors = [torch.empty((height, width, 4), **tensor_kwargs).detach() for i in range(10)]
    pc_obj_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()
    pc_cam_tensor = torch.empty((height, width, 4), **tensor_kwargs).detach()

    # render target pose
    R1 = axangle2mat((1, 0, 0), angle=0.5 * np.pi)
    R2 = axangle2mat((0, 0, 1), angle=-0.7 * np.pi)
    R = np.dot(R1, R2)
    t = np.array([-0.1, 0.1, 0.7], dtype=np.float32)
    pose = np.hstack([R, t.reshape((3, 1))])
    # pose1 = np.hstack([R, 0.1 + t.reshape((3, 1))])
    # pose2 = np.hstack([R, t.reshape((3, 1)) - 0.1])
    # pose3 = np.hstack([R, t.reshape((3, 1)) - 0.05])
    # pose4 = np.hstack([R, t.reshape((3, 1)) + 0.05])
    # renderer.set_poses([pose])

    bg_images = glob.glob("/gdrnpp_bop2022/datasets/coco/train2017/*.jpg")
    num_bg_imgs = len(bg_images)

    # rendering
    runs = 0
    t_render = 0
    # without copy to cpu, it is faster than meshrenderer: 0.0008892447471618652s 1124.549797107741fps
    # 5 objects, render instance masks: 0.0023294403235117594s 429.2876661860326fps
    # 5 objects, without instance masks: 0.0010711719353993733s 933.5569453909957fps
    # when copy to cpu: 0.002706778923670451s 369.4428057109217fps
    for j in tqdm(range(1000)):
        for obj_id, cls_name in enumerate(classes):
            t0 = time.perf_counter()
            light_pos = np.random.uniform(-0.5, 0.5, 3)
            intensity = np.random.uniform(0.8, 2)
            light_color = intensity * np.random.uniform(0.9, 1.1, 3)
            # poses = [pose, pose1, pose2, pose3, pose4]
            # obj_ids = [obj_id, obj_id, obj_id, obj_id, obj_id]
            poses = [pose]
            obj_ids = [obj_id]
            gt_extents = [extents[_obj_id] for _obj_id in obj_ids]
            # light_color = None
            # light_pos = (0, 0, 0)
            """
            bg_path = bg_images[random.randint(0, num_bg_imgs - 1)]
            bg_img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
            bg_img = cv2.resize(bg_img, (width, height))
            renderer.render(obj_ids, poses=poses,
                    image_tensor=image_tensor,
                    seg_tensor=None, rot_type='mat', pc_cam_tensor=None,
                    light_pos=light_pos, light_color=light_color,
                    extents=gt_extents,
                    background=bg_img[:,:, [2, 1, 0]])
            renderer.render(obj_ids, poses=poses,
                    image_tensor=None,
                    seg_tensor=seg_tensor, rot_type='mat', pc_cam_tensor=pc_cam_tensor,
                    light_pos=light_pos, light_color=light_color,
                    extents=None,
                    background=None)
            """
            renderer.render(
                obj_ids,
                poses=poses,
                image_tensor=image_tensor,
                seg_tensor=seg_tensor,
                rot_type="mat",
                pc_cam_tensor=pc_cam_tensor,
                pc_obj_tensor=pc_obj_tensor,
                light_pos=light_pos,
                light_color=light_color,
                extents=None,
                background=None,
            )
            for i in range(len(poses)):
                renderer.render(
                    obj_ids[i],
                    poses=poses[i],
                    image_tensor=None,
                    seg_tensor=instance_mask_tensors[i],
                    rot_type="mat",
                    pc_cam_tensor=None,
                    light_pos=None,
                    light_color=None,
                )
            im = image_tensor[:, :, :3]
            t_render += time.perf_counter() - t0
            xyz_ren = pc_obj_tensor[:, :, :3]

            runs += 1
            # torch.save(im, 'im_{}.pth'.format(cls_name))
            if False:  # show
                im = (im.cpu().numpy() + 0.5).astype(np.uint8)  # bgr
                seg = (seg_tensor[:, :, 0].cpu().numpy() * 255 + 0.5).astype(np.uint8)
                masks = [
                    (ins_mask[:, :, 0].cpu().numpy() * 255 + 0.5).astype(np.uint8)
                    for ins_mask in instance_mask_tensors[: len(poses)]
                ]
                print("seg unique: ", np.unique(seg))

                depth = pc_cam_tensor[:, :, 2].cpu().numpy()
                # depth_save = (depth * 1000).astype(np.uint16)
                # cv2.imwrite("depth_{}.png".format(cls_name), depth_save)
                img_vis = vis_image_mask_bbox_cv2(im, masks, bboxes=None, labels=None)

                xyz_np = xyz_ren.detach().cpu().numpy()
                im_points, model_points = get_img_model_points_with_coords2d(
                    seg > 0, xyz_np, coord2d, im_H=height, im_W=width
                )
                num_points = len(im_points)
                if num_points >= 4:
                    pose_est = misc.pnp_v2(
                        model_points,
                        im_points,
                        K,
                        method=cv2.SOLVEPNP_EPNP,
                        ransac=True,
                        ransac_reprojErr=3,
                        ransac_iter=100,
                    )

                    print("pose from pnp/ransac:", pose_est)
                    print("pose gt: ", pose)
                    _re, _te = calc_rt_dist_m(pose, pose_est)
                    print("re: ", _re, "te: ", _te * 100, "cm")

                xyz_show = get_emb_show(xyz_np)
                print("xyz: ", "min:", xyz_np.min(), "max: ", xyz_np.max())

                show_ims = [
                    im[:, :, [2, 1, 0]],
                    seg,
                    depth,
                    img_vis[:, :, [2, 1, 0]],
                    xyz_show,
                ]
                show_titles = [
                    f"{cls_name} color",
                    "seg",
                    "depth",
                    "inst_masks",
                    "xyz_ren",
                ]
                grid_show(show_ims, show_titles, row=2, col=3)

    print("{}s {}fps".format(t_render / runs, runs / t_render))
    renderer.close()
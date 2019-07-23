# -*- coding: utf-8 -*-
import os
import ntpath
import numpy as np
import six
from multiprocessing.dummy import Pool as ThreadPool

from CPAC.utils.nifti_utils import nifti_image_input
import nibabel as nib
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl


def read_mat(input_mat):
    if isinstance(input_mat, np.ndarray):
        mat = input_mat
    elif isinstance(input_mat, six.string_types):
        if os.path.exists(input_mat):
            mat = np.loadtxt(input_mat)
        else:
            raise IOError("ERROR norm_transformation: " + input_mat +
                          " file does not exist")
    else:
        raise TypeError("ERROR norm_transformation: input_mat should be" +
                        " either a str or a numpy.ndarray matrix")

    if mat.shape != (4, 4):
        raise ValueError("ERROR norm_transformation: the matrix should be 4x4")

    # Translation vector
    translation = mat[0:3, 3]
    # 3x3 matrice of rotation, scaling and skewing
    oth_transform = mat[0:3, 0:3]

    return translation, oth_transform


def norm_transformations(translation, oth_transform):
    tr_norm = np.linalg.norm(translation)
    affine_norm = np.linalg.norm(oth_transform - np.identity(3), 'fro')
    return pow(tr_norm, 2) + pow(affine_norm, 2)

def norm_transformation(input_mat):
    """
    Calculate the squared norm of the translation + squared Frobenium norm
    of the difference between other affine transformations and the identity
    from an fsl FLIRT transformation matrix
    Parameters
    ----------
    input_mat: str or numpy.ndarray
        Either the path to text file matrix or a matrix already imported.

    Returns
    -------
        numpy.float64
            squared norm of the translation + squared Frobenius norm of the
            difference between other affine transformations and the identity
    """
    if isinstance(input_mat, np.ndarray):
        mat = input_mat
    elif isinstance(input_mat, six.string_types):
        if os.path.exists(input_mat):
            mat = np.loadtxt(input_mat)
        else:
            raise IOError("ERROR norm_transformation: " + input_mat +
                          " file does not exist")
    else:
        raise TypeError("ERROR norm_transformation: input_mat should be" +
                        " either a str (file_path) or a numpy.ndarray matrix")

    if mat.shape != (4, 4):
        raise ValueError("ERROR norm_transformation: the matrix should be 4x4")

    # Translation vector
    translation = mat[0:3, 3]
    # 3x3 matrice of rotation, scaling and skewing
    oth_affine_transform = mat[0:3, 0:3]
    tr_norm = np.linalg.norm(translation)
    affine_norm = np.linalg.norm(oth_affine_transform - np.identity(3), 'fro')
    return pow(tr_norm, 2) + pow(affine_norm, 2)


def template_convergence(mat_file, mat_type='matrix',
                         convergence_threshold=np.finfo(np.float64).eps):
    """
    Calculate the distance between transformation matrix with a matrix of no
    transformation
    Parameters
    ----------
    mat_file: str
        path to an fsl flirt matrix
    mat_type: str
        'matrix'(default), 'ITK'
        The type of matrix used to represent the transformations
    convergence_threshold: float
        (numpy.finfo(np.float64).eps (default)) threshold for the convergence
        The threshold is how different from no transformation is the
        transformation matrix.

    Returns
    -------

    """
    if mat_type == 'matrix':
        translation, oth_transform = read_mat(mat_file)
    elif mat_type == 'ITK':
        translation, oth_transform = read_ants_mat(mat_file)
    else:
        raise ValueError("ERROR template_convergence: this matrix type does " +
                         "not exist")
    distance = norm_transformations(translation, oth_transform)

    return abs(distance) <= convergence_threshold


def create_temporary_template(img_list, out_path, avg_method='median'):
    """
    Average all the 3D images of the list into one 3D image
    WARNING---the function assumes that all the images have the same header,
    the output image will have the same header as the first image of the list
    Parameters---
    ----------
    img_list: list of str
        list of images paths
    avg_method: str
        function names from numpy library such as 'median', 'mean', 'std' ...

    Returns
    -------
    tmp_template: Nifti1Image
    """
    if not img_list:
        raise ValueError('ERROR create_temporary_template: image list is empty')

    if len(img_list) == 1:
        return img_list[0]

    # ALIGN CENTERS

    avg_data = getattr(np, avg_method)(
        np.asarray([nifti_image_input(img).get_data() for img in img_list]), 0)

    nii = nib.Nifti1Image(avg_data, nifti_image_input(img_list[0]).affine)
    nib.save(nii, out_path)
    return out_path


def register_img_list(img_list, ref_img, output_folder, dof=12,
                      interp='trilinear', cost='corratio', thread_pool=2):
    """
    Register a list of images to the reference image. The registered images are
    stored in output_folder.
    Parameters
    ----------
    img_list: list of str
        list of images paths
    ref_img: str
        path to the reference image to which the images will be registered
    output_folder: str
        path to the output directory
    dof: integer (int of long)
        number of transform degrees of freedom (FLIRT) (12 by default)
    interp: str
        ('trilinear' (default) or 'nearestneighbour' or 'sinc' or 'spline')
        final interpolation method used in reslicing
    cost: str
        ('mutualinfo' or 'corratio' (default) or 'normcorr' or 'normmi' or
         'leastsq' or 'labeldiff' or 'bbr')
        cost function
    thread_pool: int or multiprocessing.dummy.Pool
        (default 2) number of threads. You can also provide a Pool so the
        node will be added to it to be run.

    Returns
    -------
    multiple_linear_reg: list of Node
        each Node 'node' has been run and
        node.inputs.out_file contains the path to the registered image
        node.inputs.out_matrix_file contains the path to the transformation
        matrix
    """
    if not img_list:
        raise ValueError('ERROR register_img_list: image list is empty')

    # output_folder = os.getcwd()

    output_img_list = [os.path.join(output_folder, ntpath.basename(img))
                       for img in img_list]

    output_mat_list = [os.path.join(output_folder,
                                    str(ntpath.basename(img).split('.')[0])
                                    + '.mat')
                       for img in img_list]

    def flirt_node(img, out_img, out_mat):
        linear_reg = fsl.FLIRT()
        linear_reg.inputs.in_file = img
        linear_reg.inputs.out_file = out_img
        linear_reg.inputs.out_matrix_file = out_mat

        linear_reg.inputs.cost = cost
        linear_reg.inputs.dof = dof
        linear_reg.inputs.interp = interp
        linear_reg.inputs.reference = ref_img

        return linear_reg

    if isinstance(thread_pool, int):
        pool = ThreadPool(thread_pool)
    else:
        pool = thread_pool

    node_list = [flirt_node(img, out_img, out_mat)
                 for (img, out_img, out_mat) in zip(
                 img_list, output_img_list, output_mat_list)]
    pool.map(lambda node: node.run(), node_list)

    if isinstance(thread_pool, int):
        pool.close()
        pool.join()

    return node_list


def image_preproc(image, skull_strip_method='afni',
                  already_skullstripped=False, wf_name='anat_preproc'):
    from CPAC.anat_preproc import create_anat_preproc
    wf = create_anat_preproc(method=skull_strip_method,
                             already_skullstripped=already_skullstripped,
                             wf_name=wf_name)

    return wf


def template_creation_flirt(img_list, output_folder,
                            init_reg=None, avg_method='median', dof=12,
                            interp='trilinear', cost='corratio',
                            mat_type='matrix',
                            convergence_threshold=np.finfo(np.float64).eps,
                            thread_pool=2):
    """

    Parameters
    ----------
    img_list: list of str
        list of images paths
    output_folder: str
        path to the output folder (the folder must already exist)
    init_reg: list of Node or str
        (default None so no initial registration is performed)
        the output of the function register_img_list with another reference
        Reuter et al. 2012 (NeuroImage) section "Improved template estimation"
        doi:10.1016/j.neuroimage.2012.02.084 recommend to use a ramdomly
        selected image from the input dataset
    avg_method: str
        function names from numpy library such as 'median', 'mean', 'std' ...
    dof: integer (int of long)
        number of transform degrees of freedom (FLIRT) (12 by default)
    interp: str
        ('trilinear' (default) or 'nearestneighbour' or 'sinc' or 'spline')
        final interpolation method used in reslicing
    cost: str
        ('mutualinfo' or 'corratio' (default) or 'normcorr' or 'normmi' or
         'leastsq' or 'labeldiff' or 'bbr')
        cost function
    mat_type: str
        'matrix'(default), 'ITK'
        The type of matrix used to represent the transformations
    convergence_threshold: float
        (numpy.finfo(np.float64).eps (default)) threshold for the convergence
        The threshold is how different from no transformation is the
        transformation matrix.
    thread_pool: int or multiprocessing.dummy.Pool
        (default 2) number of threads. You can also provide a Pool so the
        node will be added to it to be run.

    Returns
    -------
    template: str
        path to the final template

    """
    if isinstance(thread_pool, int):
        pool = ThreadPool(thread_pool)
    else:
        pool = thread_pool

    if not img_list:
        print('ERROR create_temporary_template: image list is empty')

    if init_reg is not None:
        if isinstance(init_reg, list):
            image_list = [node.inputs.out_file for node in init_reg]
            mat_list = [node.inputs.out_matrix_file for node in init_reg]
            # test if every transformation matrix has reached the convergence
            convergence_list = [template_convergence(
                mat, mat_type, convergence_threshold) for mat in mat_list]
            converged = all(convergence_list)
        if isinstance(init_reg, six.string_types):
            if init_reg == 'center':
                print('todo')
    else:
        image_list = img_list
        converged = False

    tmp_template = os.path.join(output_folder, 'tmp_template.nii.gz')

    while not converged:
        tmp_template = create_temporary_template(image_list,
                                                 out_path=tmp_template,
                                                 avg_method=avg_method)
        reg_list_node = register_img_list(image_list,
                                          ref_img=tmp_template,
                                          output_folder=output_folder,
                                          dof=dof,
                                          interp=interp,
                                          cost=cost)

        image_list = [node.inputs.out_file for node in reg_list_node]
        mat_list = [node.inputs.out_matrix_file for node in reg_list_node]
        print(str(mat_list))
        # test if every transformation matrix has reached the convergence
        convergence_list = [template_convergence(
            mat, mat_type, convergence_threshold) for mat in mat_list]
        converged = all(convergence_list)

    if isinstance(thread_pool, int):
        pool.close()
        pool.join()

    template = tmp_template
    return template


def longitudinal_template(img_list, output_folder,
                          init_reg=None, avg_method='median', dof=12,
                          interp='trilinear', cost='corratio',
                          mat_type='matrix',
                          convergence_threshold=np.finfo(np.float64).eps,
                          thread_pool=2, method='flirt'):
    """

    Parameters
    ----------
    img_list
    output_folder
    init_reg
    avg_method
    dof
    interp
    cost
    mat_type
    convergence_threshold
    thread_pool
    method

    Returns
    -------

    Algorithm
    ---------
    Homogenize the images:
    -deobliquing
    -reorienting
    -resampling to the highest resolution of the dataset
    -aligning all the images together (setting the center of mass of the brain
    at 0,0,0)
    Selection of the first target to register the images:
    -either select an image randomly (faster) or average (default median) the
    images and use this average as target.


    """
    if method == 'flirt':
        template = template_creation_flirt(img_list, output_folder,
                                           init_reg, avg_method, dof,
                                           interp, cost,
                                           mat_type,
                                           convergence_threshold,
                                           thread_pool)
    else:
        raise ValueError(str(method)
                         + 'this method has not yet been implemented')

    return template

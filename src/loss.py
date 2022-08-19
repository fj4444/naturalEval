###
# reference: https://github.com/xwshi/NRVQA/blob/b502a681cc5522090e0eb303f6e481470f3ba1f0/piqe.py
# https://github.com/xwshi/NRVQA/blob/b502a681cc5522090e0eb303f6e481470f3ba1f0/niqe.py
# https://github.com/xwshi/NRVQA/blob/b502a681cc5522090e0eb303f6e481470f3ba1f0/brisque.py
###

import numpy as np
import cv2
from scipy.special import gamma
import math
from os.path import dirname, join
import cv2
import numpy as np
import scipy
import scipy.io
import scipy.misc
import scipy.ndimage
import scipy.special
from PIL import Image

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)

def calculate_mscn(dis_image):
    dis_image = dis_image.astype(np.float32)  # 类型转换十分重要
    ux = cv2.GaussianBlur(dis_image, (7, 7), 7/6)
    ux_sq = ux*ux
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(dis_image**2, (7, 7), 7/6)-ux_sq))

    mscn = (dis_image-ux)/(1+sigma)

    return mscn

# Function to segment block edges


def segmentEdge(blockEdge, nSegments, blockSize, windowSize):
    # Segment is defined as a collection of 6 contiguous pixels in a block edge
    segments = np.zeros((nSegments, windowSize))
    for i in range(nSegments):
        segments[i, :] = blockEdge[i:windowSize]
        if(windowSize <= (blockSize+1)):
            windowSize = windowSize+1

    return segments


def noticeDistCriterion(Block, nSegments, blockSize, windowSize, blockImpairedThreshold, N):
    # Top edge of block
    topEdge = Block[0, :]
    segTopEdge = segmentEdge(topEdge, nSegments, blockSize, windowSize)

    # Right side edge of block
    rightSideEdge = Block[:, N-1]
    rightSideEdge = np.transpose(rightSideEdge)
    segRightSideEdge = segmentEdge(
        rightSideEdge, nSegments, blockSize, windowSize)

    # Down side edge of block
    downSideEdge = Block[N-1, :]
    segDownSideEdge = segmentEdge(
        downSideEdge, nSegments, blockSize, windowSize)

    # Left side edge of block
    leftSideEdge = Block[:, 0]
    leftSideEdge = np.transpose(leftSideEdge)
    segLeftSideEdge = segmentEdge(
        leftSideEdge, nSegments, blockSize, windowSize)

    # Compute standard deviation of segments in left, right, top and down side edges of a block
    segTopEdge_stdDev = np.std(segTopEdge, axis=1)
    segRightSideEdge_stdDev = np.std(segRightSideEdge, axis=1)
    segDownSideEdge_stdDev = np.std(segDownSideEdge, axis=1)
    segLeftSideEdge_stdDev = np.std(segLeftSideEdge, axis=1)

    # Check for segment in block exhibits impairedness, if the standard deviation of the segment is less than blockImpairedThreshold.
    blockImpaired = 0
    for segIndex in range(segTopEdge.shape[0]):
        if((segTopEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segRightSideEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segDownSideEdge_stdDev[segIndex] < blockImpairedThreshold) or
                (segLeftSideEdge_stdDev[segIndex] < blockImpairedThreshold)):
            blockImpaired = 1
            break

    return blockImpaired


def noiseCriterion(Block, blockSize, blockVar):
    # Compute block standard deviation[h,w,c]=size(I)
    blockSigma = np.sqrt(blockVar)
    # Compute ratio of center and surround standard deviation
    cenSurDev = centerSurDev(Block, blockSize)
    # Relation between center-surround deviation and the block standard deviation
    blockBeta = (abs(blockSigma-cenSurDev))/(max(blockSigma, cenSurDev))

    return blockSigma, blockBeta

# Function to compute center surround Deviation of a block


def centerSurDev(Block, blockSize):
    # block center
    center1 = int((blockSize+1)/2)-1
    center2 = center1+1
    center = np.vstack((Block[:, center1], Block[:, center2]))
    # block surround
    Block = np.delete(Block, center1, axis=1)
    Block = np.delete(Block, center1, axis=1)

    # Compute standard deviation of block center and block surround
    center_std = np.std(center)
    surround_std = np.std(Block)

    # Ratio of center and surround standard deviation
    cenSurDev = (center_std/surround_std)

    # Check for nan's
    # if(isnan(cenSurDev)):
    #     cenSurDev = 0

    return cenSurDev


def piqe(im):
    blockSize = 16  # Considered 16x16 block size for overall analysis
    activityThreshold = 0.1  # Threshold used to identify high spatially prominent blocks
    blockImpairedThreshold = 0.1  # Threshold identify blocks having noticeable artifacts
    windowSize = 6  # Considered segment size in a block edge.
    nSegments = blockSize-windowSize+1  # Number of segments for each block edge
    distBlockScores = 0  # Accumulation of distorted block scores
    NHSA = 0  # Number of high spatial active blocks.

    # pad if size is not divisible by blockSize
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    originalSize = im.shape
    rows, columns = originalSize
    rowsPad = rows % blockSize
    columnsPad = columns % blockSize
    isPadded = False
    if(rowsPad > 0 or columnsPad > 0):
        if rowsPad > 0:
            rowsPad = blockSize-rowsPad
        if columnsPad > 0:
            columnsPad = blockSize-columnsPad
        isPadded = True
        padSize = [rowsPad, columnsPad]
    im = np.pad(im, ((0, rowsPad), (0, columnsPad)), 'edge')

    # Normalize image to zero mean and ~unit std
    # used circularly-symmetric Gaussian weighting function sampled out
    # to 3 standard deviations.
    imnorm = calculate_mscn(im)

    # Preallocation for masks
    NoticeableArtifactsMask = np.zeros(imnorm.shape)
    NoiseMask = np.zeros(imnorm.shape)
    ActivityMask = np.zeros(imnorm.shape)

    # Start of block by block processing
    total_var = []
    total_bscore = []
    total_ndc = []
    total_nc = []

    BlockScores = []
    for i in np.arange(0, imnorm.shape[0]-1, blockSize):
        for j in np.arange(0, imnorm.shape[1]-1, blockSize):
             # Weights Initialization
            WNDC = 0
            WNC = 0

            # Compute block variance
            Block = imnorm[i:i+blockSize, j:j+blockSize]
            blockVar = np.var(Block)

            if(blockVar > activityThreshold):
                ActivityMask[i:i+blockSize, j:j+blockSize] = 1
                NHSA = NHSA+1

                # Analyze Block for noticeable artifacts
                blockImpaired = noticeDistCriterion(
                    Block, nSegments, blockSize-1, windowSize, blockImpairedThreshold, blockSize)

                if(blockImpaired):
                    WNDC = 1
                    NoticeableArtifactsMask[i:i +
                                            blockSize, j:j+blockSize] = blockVar

                # Analyze Block for guassian noise distortions
                [blockSigma, blockBeta] = noiseCriterion(
                    Block, blockSize-1, blockVar)

                if((blockSigma > 2*blockBeta)):
                    WNC = 1
                    NoiseMask[i:i+blockSize, j:j+blockSize] = blockVar

                # Pooling/ distortion assigment
                # distBlockScores = distBlockScores + \
                #     WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2)

                if WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2) > 0:
                    BlockScores.append(
                        WNDC*pow(1-blockVar, 2) + WNC*pow(blockVar, 2))

                total_var = [total_var, blockVar]
                total_bscore = [total_bscore, WNDC *
                                (1-blockVar) + WNC*(blockVar)]
                total_ndc = [total_ndc, WNDC]
                total_nc = [total_nc, WNC]

    BlockScores = sorted(BlockScores)
    lowSum = sum(BlockScores[:int(0.1*len(BlockScores))])
    Sum = sum(BlockScores)
    Scores = [(s*10*lowSum)/Sum for s in BlockScores]
    C = 1
    Score = ((sum(Scores) + C)/(C + NHSA))*100

    # if input image is padded then remove those portions from ActivityMask,
    # NoticeableArtifactsMask and NoiseMask and ensure that size of these masks
    # are always M-by-N.
    if(isPadded):
        NoticeableArtifactsMask = NoticeableArtifactsMask[0:originalSize[0],
                                                          0:originalSize[1]]
        NoiseMask = NoiseMask[0:originalSize[0], 0:originalSize[1]]
        ActivityMask = ActivityMask[0:originalSize[0], 1:originalSize[1]]

    return Score, NoticeableArtifactsMask, NoiseMask, ActivityMask

def aggd_features(imdata):
    # flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata*imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
        gamma_hat = left_mean_sqrt/right_mean_sqrt
    else:
        gamma_hat = np.inf
    # solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
    else:
        r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) *
                          (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    # solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm)**2)
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0/alpha)
    gam2 = scipy.special.gamma(2.0/alpha)
    gam3 = scipy.special.gamma(3.0/alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    # mean parameter
    N = (br - bl)*(gam2 / gam1)  # *aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)


def ggd_features(imdata):
    nr_gam = 1/prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq/E**2
    pos = np.argmin(np.abs(nr_gam - rho))
    return gamma_range[pos], sigma_sq


def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights


def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0/6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1,
                              mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0,
                              var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window,
                              1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    return (image - mu_image)/(var_image + C), var_image, mu_image


def _niqe_extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl+br)/2.0,
                     alpha1, N1, bl1, br1,  # (V)
                     alpha2, N2, bl2, br2,  # (H)
                     alpha3, N3, bl3, bl3,  # (D1)
                     alpha4, N4, bl4, bl4,  # (D2)
                     ])


def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)


def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)


def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = np.int(patch_size)
    patches = []
    for j in range(0, h-patch_size+1, patch_size):
        for i in range(0, w-patch_size+1, patch_size):
            patch = img[j:j+patch_size, i:i+patch_size]
            patches.append(patch)

    patches = np.array(patches)

    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features


def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)

    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]

    img = img.astype(np.float32)
    # img2 = scipy.misc.imresize(img, 0.5, interp='bicubic', mode='F')
    img2 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)

    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size/2)

    feats = np.hstack((feats_lvl1, feats_lvl2))  # feats_lvl3))

    return feats


def niqe(inputImgData):

    patch_size = 96
    module_path = dirname(__file__)

    # TODO: memoize
    params = scipy.io.loadmat(
        join(module_path, 'niqe_image_params.mat'))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]

    if inputImgData.ndim == 3:
        inputImgData = cv2.cvtColor(inputImgData, cv2.COLOR_BGR2GRAY)
    M, N = inputImgData.shape

    # assert C == 1, "niqe called with videos containing %d channels. Please supply only the luminance channel" % (C,)
    assert M > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert N > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"

    feats = get_patches_test_features(inputImgData, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = ((pop_cov+sample_cov)/2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe_score

def extract_brisque_feats(mscncoefs):
    alpha_m, sigma_sq = ggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    # print(alpha_m, alpha1)
    return [
        alpha_m, sigma_sq,
        alpha1, N1, lsq1**2, rsq1**2,  # (V)
        alpha2, N2, lsq2**2, rsq2**2,  # (H)
        alpha3, N3, lsq3**2, rsq3**2,  # (D1)
        alpha4, N4, lsq4**2, rsq4**2,  # (D2)
    ]

def brisque(im):
    mscncoefs = calculate_mscn(im)
    features1 = extract_brisque_feats(mscncoefs)
    lowResolution = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
    features2 = extract_brisque_feats(lowResolution)

    return np.array(features1+features2)
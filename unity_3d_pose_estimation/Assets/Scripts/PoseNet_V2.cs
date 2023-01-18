using Unity.Barracuda;
using UnityEngine;
using System.Collections.Generic;
using UnityEngine.Video;
using UnityEngine.UI;
using System.Collections;

public class PoseNet_V2 : MonoBehaviour
{
    [Tooltip("The input image that will be fed to the model")]
    public RenderTexture videoTexture;

    [Tooltip("The ComputeShader that will perform the model-specific preprocessing")]
    public ComputeShader posenetShader;

    [Tooltip("The requested webcam height")]
    public int webcamHeight = 720;

    [Tooltip("The requested webcam width")]
    public int webcamWidth = 1280;

    [Tooltip("The requested webcam frame rate")]
    public int webcamFPS = 60;

    [Tooltip("The height of the image being fed to the model")]
    public int imageHeight = 360;

    [Tooltip("The width of the image being fed to the model")]
    public int imageWidth = 360;

    [Tooltip("Turn the InputScreen on or off")]
    public bool displayInput = false;

    [Tooltip("Use webcam feed as input")]
    public bool useWebcam = false;

    [Tooltip("The screen for viewing preprocessed images")]
    public GameObject inputScreen;

    [Tooltip("Stores the preprocessed image")]
    public RenderTexture inputTexture;

    [Tooltip("The model asset file to use when performing inference")]
    public NNModel modelAsset;

    public NNModel liftmodelAsset;

    [Tooltip("The backend to use when performing inference")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    [Tooltip("The minimum confidence level required to display the key point")]
    [Range(0, 100)]
    public int minConfidence = 70;

    [Tooltip("The list of key point GameObjects that make up the pose skeleton")]
    public GameObject[] keypoints;

    public GameObject VideoScreen;

    public LayerMask _layer;

    // The compiled model used for performing inference
    private Model m_RunTimeModel;

    private Model LiftModel;

    // The interface used to execute the neural network
    private IWorker engine;

    private IWorker engine2;

    // The name for the heatmap layer in the model asset
    private string heatmapLayer = "520";

    // The name for the offsets layer in the model asset
    private string offsetsLayer = "float_short_offsets";

    // The name for the Sigmoid layer that returns the heatmap predictions
    private string predictionLayer = "520";

    // The number of key points estimated by the model
    private const int numKeypoints = 17;

    // Stores the current estimated 2D keypoint locations in videoTexture
    // and their associated confidence values
    private float[][] keypointLocations = new float[numKeypoints][];
    // Live video input from a webcam
    private WebCamTexture webcamTexture;

    // The height of the current video source
    private int videoHeight;

    // The width of the current video source
    private int videoWidth;

    private float[][] newKeyPoint;
    private RenderTexture MainTexture;

    public List<GameObject> PoseEstimator;
    // Start is called before the first frame update
    public static int Frames = 243;
    private void Awake()
    {
        for (int i = 0; i < Frames; i++)
        {
            PoseData_X.Add(new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 });
            PoseData_Y.Add(new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
        }
    }
    void Start()
    {
        // Get a reference to the Video Player GameObject
        GameObject videoPlayer = GameObject.Find("Video Player");
        // Get the Transform component for the VideoScreen GameObject
        Transform videoScreen = GameObject.Find("VideoScreen").transform;
        if (useWebcam)
        {
            // Create a new WebCamTexture
            webcamTexture = new WebCamTexture(webcamWidth, webcamHeight, webcamFPS);

            // Flip the VideoScreen around the Y-Axis
            videoScreen.rotation = Quaternion.Euler(0, 180, 0);
            // Invert the scale value for the Z-Axis
            videoScreen.localScale = new Vector3(videoScreen.localScale.x, videoScreen.localScale.y, -1f);

            // Start the Camera
            webcamTexture.Play();

            // Deactivate the Video Player
            videoPlayer.SetActive(false);

            // Update the videoHeight
            videoHeight = (int)webcamTexture.height;
            // Update the videoWidth
            videoWidth = (int)webcamTexture.width;

        }
        else
        {
            // Update the videoHeight
            videoHeight = (int)videoPlayer.GetComponent<VideoPlayer>().height;
            // Update the videoWidth
            videoWidth = (int)videoPlayer.GetComponent<VideoPlayer>().width;
        }

        // Release the current videoTexture
        videoTexture.Release();
        // Create a new videoTexture using the current video dimensions
        videoTexture = new RenderTexture(videoWidth, videoHeight, 24, RenderTextureFormat.ARGB32);

        // Use new videoTexture for Video Player
        videoPlayer.GetComponent<VideoPlayer>().targetTexture = videoTexture;

        // Apply the new videoTexture to the VideoScreen Gameobject
        videoScreen.gameObject.GetComponent<MeshRenderer>().material.SetTexture("_MainTex", videoTexture);
        // Adjust the VideoScreen dimensions for the new videoTexture
        videoScreen.localScale = new Vector3(videoWidth, videoHeight, videoScreen.localScale.z);
        // Adjust the VideoScreen position for the new videoTexture
        videoScreen.position = new Vector3(videoWidth / 2, videoHeight / 2, 1);

        // Get a reference to the Main Camera GameObject
        GameObject mainCamera = GameObject.Find("Main Camera");
        // Adjust the camera position to account for updates to the VideoScreen
        mainCamera.transform.position = new Vector3(videoWidth / 2, videoHeight / 2, -(videoWidth / 2));
        // Adjust the camera size to account for updates to the VideoScreen
        mainCamera.GetComponent<Camera>().orthographicSize = videoHeight / 2;

        // Compile the model asset into an object oriented representation
        m_RunTimeModel = ModelLoader.Load(modelAsset);
        ModelBuilder modelbuilder = new ModelBuilder(m_RunTimeModel);
        LiftModel = ModelLoader.Load(liftmodelAsset);
        ModelBuilder liftmodelbuilder = new ModelBuilder(LiftModel);
        modelbuilder.Sigmoid(predictionLayer, m_RunTimeModel.outputs[0]);
        // Create a model builder to modify the m_RunTimeModel
        //var modelBuilder = new ModelBuilder(m_RunTimeModel);
        // Add a new Sigmoid layer that takes the output of the heatmap layer
        //modelBuilder.Sigmoid(predictionLayer, heatmapLayer);

        // Create a worker that will execute the model with the selected backend
        engine = WorkerFactory.CreateWorker(workerType, modelbuilder.model);
        engine2 = WorkerFactory.CreateWorker(workerType, liftmodelbuilder.model);
        //engine = WorkerFactory.CreateWorker(workerType, m_RunTimeModel);
        CameraSet();

    }

    // OnDisable is called when the MonoBehavior becomes disabled or inactive
    private void OnDisable()
    {
        // Release the resources allocated for the inference engine
        engine.Dispose();
        engine2.Dispose();
    }
    private void LateUpdate()
    {
        Lifting();

    }
    // Update is called once per frame
    void Update()
    {
        if (useWebcam)
        {
            // Copy webcamTexture to videoTexture
            Graphics.Blit(webcamTexture, videoTexture);
        }


        // Preprocess the image for the current frame
        Texture2D processedImage = PreprocessImage();

        if (displayInput)
        {
            // Activate the InputScreen GameObject
            inputScreen.SetActive(true);
            // Create a temporary Texture2D to store the rescaled input image
            Texture2D scaledInputImage = ScaleInputImage(processedImage);
            // Copy the data from the Texture2D to the RenderTexture
            Graphics.Blit(scaledInputImage, inputTexture);
            // Destroy the temporary Texture2D
            Destroy(scaledInputImage);
        }
        else
        {
            // Deactivate the InputScreen GameObject
            inputScreen.SetActive(false);
        }

        /*videoTexture = imageTexture;
        // Create a Tensor of shape [1, processedImage.height, processedImage.width, 3]
         Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>() { { "input.1", null }, { "input.4", null }, { "input.7", null }, };
         Tensor input = new Tensor(imageTexture);
         inputs["input.1"] = input; 
         inputs["input.4"] = new Tensor(imageTexture); 
         inputs["input.7"] = new Tensor(imageTexture); 
         */
        //one input
        Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>() { { "input.1", null } };
        inputs["input.1"] = new Tensor(MainTexture, channels: 3);
        // Execute neural network with the provided input
        engine.Execute(inputs);


        // Determine the key point locations
        ProcessOutput(engine.PeekOutput(m_RunTimeModel.outputs[0]));
        // Update the positions for the key point GameObjects
        UpdateKeyPointPositions();

        // Release GPU resources allocated for the Tensor
        inputs["input.1"].Dispose();
        // Remove the processedImage variable
        Destroy(processedImage);

    }
    private void CameraSet()
    {
        GameObject go = new GameObject("MainTextureCamera", typeof(Camera));

        go.transform.parent = VideoScreen.transform;
        go.transform.localScale = new Vector3(1.0f, -1.0f, 1.0f);
        go.transform.localPosition = new Vector3(0.0f, 0.0f, -2.0f);
        go.transform.localEulerAngles = new Vector3(0f, 180f, 0f);
        go.layer = _layer;

        var camera = go.GetComponent<Camera>();
        camera.orthographic = true;
        camera.orthographicSize = 360f;
        camera.depth = -5;
        camera.depthTextureMode = 0;
        camera.clearFlags = CameraClearFlags.Color;
        camera.backgroundColor = Color.black;
        camera.cullingMask = _layer;
        camera.useOcclusionCulling = false;
        camera.nearClipPlane = 1.0f;
        camera.farClipPlane = 5.0f;
        camera.allowMSAA = false;
        camera.allowHDR = false;
        MainTexture = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.ARGBHalf);
        camera.targetTexture = MainTexture;
    }
    /// <summary>
    /// Prepare the image to be fed into the neural network
    /// </summary>
    /// <returns>The processed image</returns>
    private Texture2D PreprocessImage()
    {
        // Create a new Texture2D with the same dimensions as videoTexture
        Texture2D imageTexture = new Texture2D(videoTexture.width, videoTexture.height, TextureFormat.RGBA32, false);

        // Copy the RenderTexture contents to the new Texture2D
        Graphics.CopyTexture(videoTexture, imageTexture);

        // Make a temporary Texture2D to store the resized image
        Texture2D tempTex = Resize(imageTexture, imageHeight, imageWidth);
        // Remove the original imageTexture
        Destroy(imageTexture);

        // Apply model-specific preprocessing
        imageTexture = PreprocessResNet(tempTex);
        // Remove the temporary Texture2D
        Destroy(tempTex);

        return imageTexture;
    }

    /// <summary>
    /// Resize the provided Texture2D
    /// </summary>
    /// <param name="image">The image to be resized</param>
    /// <param name="newWidth">The new image width</param>
    /// <param name="newHeight">The new image height</param>
    /// <returns>The resized image</returns>
    private Texture2D Resize(Texture2D image, int newWidth, int newHeight)
    {
        // Create a temporary RenderTexture
        RenderTexture rTex = RenderTexture.GetTemporary(newWidth, newHeight, 24);
        // Make the temporary RenderTexture the active RenderTexture
        RenderTexture.active = rTex;

        // Copy the Texture2D to the temporary RenderTexture
        Graphics.Blit(image, rTex);
        // Create a new Texture2D with the new Dimensions
        Texture2D nTex = new Texture2D(newWidth, newHeight, TextureFormat.RGBA32, false);

        // Copy the temporary RenderTexture to the new Texture2D
        Graphics.CopyTexture(rTex, nTex);

        // Make the temporary RenderTexture not the active RenderTexture
        RenderTexture.active = null;

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(rTex);
        return nTex;
    }

    /// <summary>
    /// Perform model-specific preprocessing on the GPU
    /// </summary>
    /// <param name="inputImage">The image to be processed</param>
    /// <returns>The processed image</returns>
    private Texture2D PreprocessResNet(Texture2D inputImage)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the PreprocessResNet function in the ComputeShader
        //int kernelHandle = posenetShader.FindKernel("PreprocessResNet");
        // Define an HDR RenderTexture
        RenderTexture rTex = new RenderTexture(inputImage.width, inputImage.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        rTex.enableRandomWrite = true;
        // Create the HDR RenderTexture
        rTex.Create();

        // Set the value for the Result variable in the ComputeShader
        //posenetShader.SetTexture(kernelHandle, "Result", rTex);
        // Set the value for the InputImage variable in the ComputeShader
        //posenetShader.SetTexture(kernelHandle, "InputImage", inputImage);

        // Execute the ComputeShader
        //posenetShader.Dispatch(kernelHandle, inputImage.height / numthreads, inputImage.width / numthreads, 1);
        // Make the HDR RenderTexture the active RenderTexture
        RenderTexture.active = rTex;

        // Create a new HDR Texture2D
        Texture2D nTex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBAHalf, false);

        // Copy the RenderTexture to the new Texture2D
        Graphics.CopyTexture(rTex, nTex);
        // Make the HDR RenderTexture not the active RenderTexture
        RenderTexture.active = null;
        // Remove the HDR RenderTexture
        Destroy(rTex);
        return nTex;
    }

    /// <summary>
    /// Rescale the pixel values from [0, 255] to [0.0, 1.0]
    /// </summary>
    /// <param name="inputImage"></param>
    /// <returns>The rescaled image</returns>
    private Texture2D ScaleInputImage(Texture2D inputImage)
    {
        // Specify the number of threads on the GPU
        int numthreads = 8;
        // Get the index for the ScaleInputImage function in the ComputeShader
        int kernelHandle = posenetShader.FindKernel("ScaleInputImage");
        // Define an HDR RenderTexture
        RenderTexture rTex = new RenderTexture(inputImage.width, inputImage.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        rTex.enableRandomWrite = true;
        // Create the HDR RenderTexture
        rTex.Create();

        // Set the value for the Result variable in the ComputeShader
        posenetShader.SetTexture(kernelHandle, "Result", rTex);
        // Set the value for the InputImage variable in the ComputeShader
        posenetShader.SetTexture(kernelHandle, "InputImage", inputImage);

        // Execute the ComputeShader
        posenetShader.Dispatch(kernelHandle, inputImage.height / numthreads, inputImage.width / numthreads, 1);
        // Make the HDR RenderTexture the active RenderTexture
        RenderTexture.active = rTex;

        // Create a new HDR Texture2D
        Texture2D nTex = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBAHalf, false);

        // Copy the RenderTexture to the new Texture2D
        Graphics.CopyTexture(rTex, nTex);
        // Make the HDR RenderTexture not the active RenderTexture
        RenderTexture.active = null;
        // Remove the HDR RenderTexture
        Destroy(rTex);
        return nTex;
    }

    /// <summary>
    /// Determine the estimated key point locations using the heatmaps and offsets tensors
    /// </summary>
    /// <param name="heatmaps">The heatmaps that indicate the confidence levels for key point locations</param>
    /// <param name="offsets">The offsets that refine the key point locations determined with the heatmaps</param>
    private void ProcessOutput(Tensor heatmaps)
    {
        // Calculate the stride used to scale down the inputImage
        float stride = 5;

        // The smallest dimension of the videoTexture
        int minDimension = Mathf.Min(videoTexture.width, videoTexture.height);
        // The largest dimension of the videoTexture
        int maxDimension = Mathf.Max(videoTexture.width, videoTexture.height);

        // The value used to scale the key point locations up to the source resolution
        float scale = (float)minDimension / (float)Mathf.Min(imageWidth, imageHeight);
        // The value used to compensate for resizing the source image to a square aspect ratio
        float unsqueezeScale = (float)maxDimension / (float)minDimension;
        float[] what = heatmaps.data.Download(heatmaps.shape);
        // Iterate through heatmaps
        for (int k = 0; k < numKeypoints; k++)
        {
            // Get the location of the current key point and its associated confidence value
            var locationInfo = LocateKeyPointIndex(heatmaps, k);

            // The (x, y) coordinates containing the confidence value in the current heatmap
            var coords = locationInfo.Item1;
            // The accompanying offset vector for the current coords
            // The associated confidence value
            var confidenceValue = locationInfo.Item2;
            // Calcluate the X-axis position
            // Scale the X coordinate up to the inputImage resolution
            // Add the offset vector to refine the key point location
            // Scale the position up to the videoTexture resolution
            // Compensate for any change in aspect ratio
            float xPos = (coords[0] * stride) * scale;

            // Calculate the Y-axis position
            // Scale the Y coordinate up to the inputImage resolution and subtract it from the imageHeight
            // Add the offset vector to refine the key point location
            // Scale the position up to the videoTexture resolution
            float yPos = (imageHeight - (coords[1] * stride)) * scale;
            //Debug.Log($"{coords[0]}, {coords[1]},{xPos}, {yPos}");
            if (videoTexture.width > videoTexture.height)
            {
                xPos *= unsqueezeScale;
            }
            else
            {
                yPos *= unsqueezeScale;
            }

            // Flip the x position if using a webcam
            if (useWebcam)
            {
                xPos = videoTexture.width - xPos;
            }
            // Update the estimated key point location in the source image
            keypointLocations[k] = new float[] { xPos, yPos, confidenceValue };
        }
        heatmaps.Dispose();
    }

    /// <summary>
    /// Find the heatmap index that contains the highest confidence value and the associated offset vector
    /// </summary>
    /// <param name="heatmaps"></param>
    /// <param name="offsets"></param>
    /// <param name="keypointIndex"></param>
    /// <returns>The heatmap index, offset vector, and associated confidence value</returns>
    private (float[], float) LocateKeyPointIndex(Tensor heatmaps, int keypointIndex)
    {
        // Stores the highest confidence value found in the current heatmap
        float maxConfidence = 0f;

        // The (x, y) coordinates containing the confidence value in the current heatmap
        float[] coords = new float[2];
        // The accompanying offset vector for the current coords


        // Iterate through heatmap columns
        for (int y = 0; y < heatmaps.height; y++)
        {
            // Iterate through column rows
            for (int x = 0; x < heatmaps.width; x++)
            {
                if (heatmaps[0, y, x, keypointIndex] > maxConfidence)
                {
                    //Debug.Log("x: " + x + "y: " + y);
                    // Update the highest confidence for the current key point
                    maxConfidence = heatmaps[0, y, x, keypointIndex];

                    // Update the estimated key point coordinates
                    coords = new float[] { x, y };
                }
            }

        }
        return (coords, maxConfidence);
    }
    /// <summary>
    /// Update the positions for the key point GameObjects
    /// </summary>
    private void UpdateKeyPointPositions()
    {
        // Iterate through the key points
        for (int k = 0; k < numKeypoints; k++)
        {
            // Check if the current confidence value meets the confidence threshold
            if (keypointLocations[k][2] >= minConfidence / 100f)
            {
                // Activate the current key point GameObject
                keypoints[k].SetActive(true);
                //keypoints[k].activeInHierarchy
            }
            else
            {
                // Deactivate the current key point GameObject
                keypoints[k].SetActive(false);
            }

            // Create a new position Vector3
            // Set the z value to -1f to place it in front of the video screen
            Vector3 newPos = new Vector3(keypointLocations[k][0], keypointLocations[k][1], -1f);

            // Update the current key point location
            keypoints[k].transform.position = newPos;
        }
    }
    List<float[]> PoseData_X = new List<float[]>(Frames);
    List<float[]> PoseData_Y = new List<float[]>(Frames);
    TensorShape input = new TensorShape(1, 17, 2, Frames);
    float[][] ThreeDpos = new float[17][];
    private void Lifting()
    {
        Tensor InputTensor = new Tensor(input);
        float[][] CalKeypoint = new float[17][];

        CalKeypoint[0] = new float[] { (keypointLocations[11][0] + keypointLocations[12][0]) / 2, (keypointLocations[11][1] + keypointLocations[12][1]) / 2 };
        CalKeypoint[8] = new float[] { (keypointLocations[5][0] + keypointLocations[6][0]) / 2, (keypointLocations[5][1] + keypointLocations[6][1]) / 2 };

        CalKeypoint[10] = new float[] { (keypointLocations[0][0] * 2 - CalKeypoint[8][0]), (keypointLocations[0][1] * 2 - CalKeypoint[8][1]) };
        CalKeypoint[7] = new float[] { (CalKeypoint[0][0] + CalKeypoint[8][0]) / 2, (CalKeypoint[0][1] + CalKeypoint[8][1]) / 2 };
        CalKeypoint[1] = new float[] { keypointLocations[12][0], keypointLocations[12][1] };
        CalKeypoint[2] = new float[] { keypointLocations[14][0], keypointLocations[14][1] };
        CalKeypoint[3] = new float[] { keypointLocations[16][0], keypointLocations[16][1] };
        CalKeypoint[4] = new float[] { keypointLocations[11][0], keypointLocations[11][1] };
        CalKeypoint[5] = new float[] { keypointLocations[13][0], keypointLocations[13][1] };
        CalKeypoint[6] = new float[] { keypointLocations[15][0], keypointLocations[15][1] };
        CalKeypoint[9] = new float[] { keypointLocations[0][0], keypointLocations[0][1] };
        CalKeypoint[11] = new float[] { keypointLocations[5][0], keypointLocations[5][1] };
        CalKeypoint[12] = new float[] { keypointLocations[7][0], keypointLocations[7][1] };
        CalKeypoint[13] = new float[] { keypointLocations[9][0], keypointLocations[9][1] };
        CalKeypoint[14] = new float[] { keypointLocations[6][0], keypointLocations[6][1] };
        CalKeypoint[15] = new float[] { keypointLocations[8][0], keypointLocations[8][1] };
        CalKeypoint[16] = new float[] { keypointLocations[10][0], keypointLocations[10][1] };
        float[] InputKey_X = new float[17];
        float[] InputKey_Y = new float[17];
        for (int i = 0; i < 17; i++)
        {
            InputKey_X[i] = CalKeypoint[i][0];
            InputKey_Y[i] = CalKeypoint[i][1];
        }
        PoseData_X.RemoveAt(0);
        PoseData_X.Add(InputKey_X);
        PoseData_Y.RemoveAt(0);
        PoseData_Y.Add(InputKey_Y);
        //Debug.Log(PoseData.Count);
        //Debug.Log("Error1");
        for (int i = 0; i < InputTensor.channels; i++)
        {
            for (int j = 0; j < InputTensor.height; j++)
            {
                InputTensor[0, j, 0, i] = PoseData_X[i][j];
                InputTensor[0, j, 1, i] = PoseData_Y[i][j];
            }
        }
        Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>() { { "input", null } };
        inputs["input"] = InputTensor;
        engine2.Execute(inputs);
        Tensor output = engine2.PeekOutput(LiftModel.outputs[0]);


        for (int i = 0; i < output.height; i++)
        {
            //Debug.Log(output.channels);
            float x, y, z;
            x = output[0, i, 0, 0];
            y = output[0, i, 1, 0];
            z = -output[0, i, 2, 0];
            //Debug.Log(x + y + z);
            ThreeDpos[i] = new float[] { x, y, z };
            Debug.Log("Pose " + i + ": " + x + y + z);
        }

        for (int i = 0; i < output.height; i++)
        {
            Vector3 OutputPos = new Vector3(ThreeDpos[i][0], ThreeDpos[i][1], ThreeDpos[i][2]);
            //PoseEstimator[i].transform.position = OutputPos;
            PoseEstimator[i].transform.localPosition = OutputPos;
            //PoseEstimator[i].transform.rotation = Quaternion.identity;
        }
        inputs["input"].Dispose();
        InputTensor.Dispose();
        output.Dispose();
    }

}

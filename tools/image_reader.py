from mcp.server.fastmcp import FastMCP
import base64

mcp = FastMCP("quip-browse-server")

@mcp.tool()
def read_image(image_path):
    '''Read a png/jpeg image from the specified path to it. Only support png/jpeg/jpg format.
    Args: 
        image_path: path to the image
    Returns:
        Bytes data of the image.
    '''
    ext = image_path.split(".")[-1].lower()
    if ext == "png":
        mime_type = "image/png"
    elif ext in ("jpg", "jpeg"):
        mime_type = "image/jpeg"
    else:
        raise ValueError("Only PNG and JPEG formats are supported.")

    with open(image_path, "rb") as img_file:
        bytes_data = img_file.read()
        
    return str({"image_bytes_data": bytes_data})


if __name__ == "__main__":
    print("image_reader mcp starts......")
    mcp.run()
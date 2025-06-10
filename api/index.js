export const config = {
  api: {
    bodyParser: true, // Ensure POST JSON body is parsed
  },
};

export default async function handler(req, res) {
  if (req.method === 'POST') {
    const data = req.body;
    return res.status(200).json({
      message: "Received",
      data: data,
    });
  }

  // For all non-POST methods
  return res.status(405).json({ detail: "Method Not Allowed" });
}

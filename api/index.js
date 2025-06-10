export default async function handler(req, res) {
    if (req.method === 'POST') {
    const data = req.body;
    return res.status(200).json({
        message: "Received",
        data: data,
    });
    }

    return res.status(404).json({ error: "Only POST requests are supported" });
}

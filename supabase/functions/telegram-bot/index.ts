import "https://deno.land/std@0.168.0/dotenv/load.ts";

const TELEGRAM_BOT_TOKEN = Deno.env.get("TELEGRAM_BOT_TOKEN") || "";
const GITHUB_REPO_URL = "https://github.com/reyanshgugnani23/Drone-CDS";

const WELCOME_MESSAGE = `🛸 *Welcome to Drone-CDS (Drone Defect Sensing)*

Drone-CDS is an automated computer-vision monitoring system designed for real-time road inspection. It detects potholes, filters out false positives like road paint and tar patches, and broadcasts live alert notifications.

🔗 *GitHub Repository:*
${GITHUB_REPO_URL}

🛠️ *Quick Setup Steps:*
1️⃣ \`git clone https://github.com/reyanshgugnani23/Drone-CDS.git\`
2️⃣ \`cd Drone-CDS\`
3️⃣ \`pip install -r requirements.txt\`
4️⃣ Add your \`.env\` with \`TELEGRAM_BOT_TOKEN\` & \`TELEGRAM_CHAT_ID\`
5️⃣ Run \`python main.py\`

⚡ *Bot Status: Active & Ready!*`;

Deno.serve(async (req: Request) => {
  // Handle CORS preflight requests
  if (req.method === "OPTIONS") {
    return new Response("ok", {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
      },
    });
  }

  try {
    const update = await req.json();
    console.log("Received Telegram update:", JSON.stringify(update));

    const text = update?.message?.text;
    const chatId = update?.message?.chat?.id;

    if ((text === "/start" || text === "/help") && chatId) {
      if (!TELEGRAM_BOT_TOKEN) {
        console.error("TELEGRAM_BOT_TOKEN is missing in environment variables!");
        return new Response(JSON.stringify({ error: "Bot token not configured" }), {
          status: 500,
          headers: { "Content-Type": "application/json" },
        });
      }

      const telegramUrl = `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`;
      const telegramRes = await fetch(telegramUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chat_id: chatId,
          text: WELCOME_MESSAGE,
          parse_mode: "Markdown",
          disable_web_page_preview: false,
        }),
      });

      const telegramData = await telegramRes.json();
      console.log("Telegram API response:", JSON.stringify(telegramData));
    }

    return new Response(JSON.stringify({ status: "ok" }), {
      headers: { "Content-Type": "application/json" },
      status: 200,
    });
  } catch (error: any) {
    console.error("Error processing request:", error.message);
    return new Response(JSON.stringify({ error: error.message }), {
      headers: { "Content-Type": "application/json" },
      status: 200, // Return 200 to Telegram so it stops retrying failed updates
    });
  }
});
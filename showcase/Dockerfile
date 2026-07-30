# RMG showcase (Next.js static export → nginx).
#
# next.config.mjs sets `output: "export"`, so `next build` emits ./out with no
# Node server at runtime — the runner stage is just nginx over static files.
FROM node:20-slim AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine AS runner
# `trailingSlash: true` means every route is a directory with an index.html,
# which nginx's default index handling already serves correctly.
COPY --from=builder /app/out /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80

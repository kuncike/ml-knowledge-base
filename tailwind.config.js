/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // 莫兰迪配色系统 — 低饱和度高级灰
        morandi: {
          // 背景色系 — 柔和燕麦色/雾霾蓝
          bg: {
            primary:   '#f4f4f2',   // 燕麦白（主背景）
            secondary: '#e8ecef',   // 雾霾蓝灰（侧边栏/卡片）
            tertiary:  '#edebe8',   // 暖灰（hover 态）
            code:      '#eef1f3',   // 代码块背景
          },
          // 文本色系
          text: {
            primary:   '#3a3f47',   // 深灰蓝（正文）
            secondary: '#6b7280',   // 中灰（辅助文字）
            muted:     '#9ca3af',   // 浅灰（占位/禁用）
          },
          // 边框色系
          border: {
            light: '#e2e4e6',
            mid:   '#d1d5db',
          },
          // 强调色 — 豆沙绿 / 灰藕粉 / 冷灰蓝
          accent: {
            green:  '#8ba09e',   // 豆沙绿（主要强调）
            pink:   '#c6b8b1',   // 灰藕粉（次要强调/标签）
            blue:   '#7b8b9a',   // 冷灰蓝（链接/图谱线条）
            purple: '#9b9ab0',   // 灰紫（高亮/特殊标记）
          },
        },
      },
      fontFamily: {
        sans: ['"Inter"', '"Noto Sans SC"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
      typography: ({ theme }) => ({
        morandi: {
          css: {
            '--tw-prose-body':        theme('colors.morandi.text.primary'),
            '--tw-prose-headings':    theme('colors.morandi.text.primary'),
            '--tw-prose-links':       theme('colors.morandi.accent.blue'),
            '--tw-prose-code':        theme('colors.morandi.accent.purple'),
            '--tw-prose-pre-bg':      theme('colors.morandi.bg.code'),
            '--tw-prose-pre-code':    theme('colors.morandi.text.primary'),
            maxWidth: 'none',
          },
        },
      }),
    },
  },
  plugins: [require('@tailwindcss/typography')],
}

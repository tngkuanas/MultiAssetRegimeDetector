// theme.js
import { extendTheme } from '@chakra-ui/react';
import { mode } from '@chakra-ui/theme-tools';

const colors = {
  brand: {
    // Backgrounds
    900: '#09090b', // Deep Zinc
    800: '#18181b', // Deep Zinc
    700: '#27272a', // Border color

    // Text
    100: '#e4e4e7', // White
    200: '#71717a', // Muted Gray

    // Accent
    red: '#dc2626',
    
    // Status
    success: '#10b981', // Emerald
    danger: '#ef4444', // Red
  },
};

const fonts = {
  heading: `'Inter', sans-serif`,
  body: `'Inter', sans-serif`,
  mono: `'Roboto Mono', monospace`,
};

const styles = {
  global: (props) => ({
    body: {
      bg: mode('white', '#000000')(props),
      color: mode('gray.800', 'brand.100')(props),
    },
  }),
};

const components = {
  Heading: {
    baseStyle: {
      fontFamily: 'heading',
      fontWeight: '900',
      textTransform: 'uppercase',
      letterSpacing: 'wide',
    },
  },
  Text: {
    baseStyle: {
      fontFamily: 'body',
    },
  },
  Button: {
    variants: {
      'solid-red': {
        bg: 'brand.red',
        color: 'white',
        _hover: {
          bg: '#b91c1c',
        },
      },
    },
  },
};

const theme = extendTheme({
  colors,
  fonts,
  styles,
  components,
});

export default theme;

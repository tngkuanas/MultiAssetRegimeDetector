import React from 'react';
import { Box } from '@chakra-ui/react';

const MainContent = ({ children }) => {
  return (
    <Box 
      ml="320px" 
      w="calc(100% - 320px)" 
      h="100vh" 
      bg="black" 
      color="brand.100" // Set default text color for children
      px={8}
      py={8}
      pb={24}
      overflowY="auto" // Enable vertical scrolling if content overflows
    >
      {children}
    </Box>
  );
};

export default MainContent;

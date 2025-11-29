import React, { useState } from 'react';
import {
  Box,
  VStack,
  Heading,
  Text,
  Icon,
  Textarea,
  FormControl,
  FormLabel,
  FormHelperText,
  Input,
  Button,
  Flex,
  useTheme,
} from '@chakra-ui/react';
import { FaMoneyBillAlt } from 'react-icons/fa';

import DatePicker from 'react-datepicker';

const Sidebar = ({ onStartBacktest, jobStatus }) => {
  const theme = useTheme();
  // Dates are now stored as Date objects for the datepicker, but passed as strings
  const [startDate, setStartDate] = useState(new Date('2018-01-01'));
  const [endDate, setEndDate] = useState(new Date('2024-01-01'));
  const [symbols, setSymbols] = useState('VOO, VXUS, GLD, BND, GOOG');

  const isLoading = jobStatus === 'in_progress';

  const handleStart = () => {
    // Format dates back to 'YYYY-MM-DD' strings for the API
    const formattedStartDate = startDate.toISOString().split('T')[0];
    const formattedEndDate = endDate.toISOString().split('T')[0];
    onStartBacktest({ symbols, startDate: formattedStartDate, endDate: formattedEndDate });
  };

  // Custom Input for the DatePicker
  const DatePickerInput = React.forwardRef(({ value, onClick }, ref) => (
    <Input
      onClick={onClick}
      ref={ref}
      value={value}
      bg="brand.800"
      borderColor="brand.700"
      color="brand.100"
      _hover={{ borderColor: 'brand.200' }}
      _focus={{ borderColor: 'brand.red', boxShadow: `0 0 0 1px ${theme.colors.brand.red}` }}
      readOnly
    />
  ));

  return (
    <Box
      as="aside"
      w="320px"
      bg="brand.900"
      p={6}
      borderRight="1px solid"
      borderColor="brand.700"
      h="100vh"
      position="fixed"
      left="0"
      top="0"
      display="flex"
      flexDirection="column"
      zIndex={20}
    >
      {/* Header */}
      <VStack align="flex-start" spacing={12} flexShrink={0}>
        <Box display="flex" alignItems="center">
          <Icon as={FaMoneyBillAlt} w={8} h={8} color="brand.red" />
          <Heading as="h1" size="md" ml={3} letterSpacing="wider">
            <Text as="span" color="white">REGIME</Text>
            <Text as="span" color="brand.red">DETECTOR</Text>
          </Heading>
        </Box>
      </VStack>

      {/* Form Inputs */}
      <VStack spacing={8} mt={12} flexGrow={1} w="full">
        <FormControl>
          <FormLabel textTransform="uppercase" fontWeight="bold" fontSize="sm" color="brand.200">
            Portfolio Universe
          </FormLabel>
          <Textarea
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            placeholder="e.g., VOO, VXUS, GLD, BND, GOOG"
            bg="brand.800"
            borderColor="brand.700"
            color="brand.100"
            fontFamily="mono"
            textTransform="uppercase"
            _hover={{ borderColor: 'brand.200' }}
            _focus={{ borderColor: 'brand.red', boxShadow: `0 0 0 1px ${theme.colors.brand.red}` }}
            _placeholder={{ color: 'brand.200' }}
          />
          <FormHelperText textAlign="right" color="brand.200">
            Comma separated
          </FormHelperText>
        </FormControl>

        <FormControl>
          <FormLabel textTransform="uppercase" fontWeight="bold" fontSize="sm" color="brand.200">
            Analysis Window
          </FormLabel>
          <Flex>
            <Box w="full">
              <DatePicker
                selected={startDate}
                onChange={(date) => setStartDate(date)}
                selectsStart
                startDate={startDate}
                endDate={endDate}
                customInput={<DatePickerInput />}
                popperPlacement="right-start"
                showYearDropdown
                scrollableYearDropdown
                yearDropdownItemNumber={10}
              />
            </Box>
            <Box w="full" ml={2}>
              <DatePicker
                selected={endDate}
                onChange={(date) => setEndDate(date)}
                selectsEnd
                startDate={startDate}
                endDate={endDate}
                minDate={startDate}
                customInput={<DatePickerInput />}
                popperPlacement="right-start"
                showYearDropdown
                scrollableYearDropdown
                yearDropdownItemNumber={10}
              />
            </Box>
          </Flex>
        </FormControl>
      </VStack>

      {/* Footer Button */}
      <Box mt="auto" flexShrink={0}>
        <Button
          variant="solid-red"
          w="full"
          py={6}
          isLoading={isLoading}
          onClick={handleStart}
          loadingText="SIMULATING STRATEGY..."
          spinnerPlacement="start"
          _loading={{
            bg: 'gray.700',
            cursor: 'wait',
          }}
          boxShadow="0 0 20px rgba(220, 38, 38, 0.4)"
        >
          RUN MODEL
        </Button>
      </Box>
    </Box>
  );
};

export default Sidebar;
